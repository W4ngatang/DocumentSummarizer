-- Modified from Yoon's seq2seq
-- Note that wherever this says "sentence", it really means "document". When it
-- says "word", it means "document". When it says "character", it means "word".
require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'models.lua'
require 'model_utils.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5',[[Path to the training *.hdf5 file 
from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file 
from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as 
savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is 
the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards, 
then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-use_chars_enc', 1, [[If 1, use character on the encoder 
side (instead of word embeddings - this is now required for the document encoder]])
cmd:option('-reverse_src', 0, [[If 1, reverse the source sequence. The original 
sequence-to-sequence paper found that this was crucial to 
achieving good performance, but with attention models this
does not seem necessary. Recommend leaving it to 0]])
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time 
0 to be the last hidden/cell state of the encoder. If 0, 
the initial states of the decoder are set to zero vectors]])
cmd:option('-hop_attn', 0, [[If > 0, then use a *hop attention* on this layer of the decoder. 
For example, if num_layers = 3 and `hop_attn = 2`, then the 
model will do an attention over the source sequence
on the second layer (and use that as input to the third layer) and 
the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to 
the l-th LSTM layer if the hidden state of the l-1-th LSTM layer 
added with the l-2th LSTM layer. We didn't find this to help in our 
experiments]])

cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

-- char-cnn model specs (if use_chars == 1)
cmd:option('-kernel_width', 6, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', 1000, [[Number of convolutional filters (feature maps). So the
representation from characters will have this many dimensions]])
cmd:option('-num_highway_layers', 2, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoing, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support
(-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it
to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. 
Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
sequence length. Sometimes setting this to 1 will increase convergence speed.]])
-- TODO: fix these
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load 
pretrained word embeddings (hdf5 file) on the encoder side. 
See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load 
pretrained word embeddings (hdf5 file) on the decoder side. 
See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")


cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
is on the first GPU and the decoder is on the second GPU. 
This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 0, [[Whether to use cudnn or not for convolutions (for the character model).
cudnn has much faster convolutions so this is highly recommended 
if using the character model]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)


function zero_table(t)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    t[i]:zero()
  end
end

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}

  for i = 1, #layers do
    if opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end      
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  if opt.pre_word_vecs_enc:len() > 0 then   
    local f = hdf5.open(opt.pre_word_vecs_enc)     
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_enc.weight[1]:copy(pre_word_vecs[i])
    end      
  end
  if opt.pre_word_vecs_dec:len() > 0 then      
    local f = hdf5.open(opt.pre_word_vecs_dec)     
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_dec.weight[1]:copy(pre_word_vecs[i])
    end      
  end

  print("Number of parameters: " .. num_params)

  if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
    cutorch.setDevice(opt.gpuid)
    word_vecs_enc.weight[1]:zero()      
    cutorch.setDevice(opt.gpuid2)
    word_vecs_dec.weight[1]:zero()
  else
    word_vecs_enc.weight[1]:zero()            
    word_vecs_dec.weight[1]:zero()
  end         

  -- prototypes for gradients so there is no need to clone
  local encoder_grad_proto = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.rnn_size)
  local encoder_grad_proto2 = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.rnn_size)
  context_proto2 = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.rnn_size)
  sent_enc_proto = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.num_kernels)
  sent_enc_grad_proto = torch.zeros(valid_data.batch_l:max(), opt.max_sent_l, opt.num_kernels) -- added by Jeffrey

  -- clone encoder/decoder up to max source/target length   
  decoder_clones = clone_many_times(decoder, opt.max_sent_l)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l)
  for i = 1, opt.max_sent_l do
    attn_clones_idx = i
    decoder_clones[i]:apply(get_layer)
    if decoder_clones[i].apply then
      decoder_clones[i]:apply(function(m) m:setReuse() end)
    end
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
    end
  end   

  local h_init_dec = torch.zeros(valid_data.batch_l:max(), opt.rnn_size)
  local h_init_enc = torch.zeros(valid_data.batch_l:max(), opt.rnn_size)      
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()      
    h_init_dec = h_init_dec:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      encoder_grad_proto2 = encoder_grad_proto2:cuda()
      context_proto = context_proto:cuda()	 
      cutorch.setDevice(opt.gpuid2)
      encoder_grad_proto = encoder_grad_proto:cuda()
      context_proto2 = context_proto2:cuda()	 
    else
      context_proto = context_proto:cuda()
      sent_enc_proto = sent_enc_proto:cuda()
      sent_enc_grad_proto = sent_enc_grad_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()	 
    end
  end

  init_fwd_enc = {}
  init_bwd_enc = {}
  init_fwd_dec = {h_init_dec:clone()} -- initial context
  init_bwd_dec = {h_init_dec:clone()} -- just need one copy of this

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_bwd_enc, h_init_enc:clone())
    table.insert(init_bwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
    table.insert(init_bwd_dec, h_init_dec:clone())
    table.insert(init_bwd_dec, h_init_dec:clone())      
  end      

  function reset_state(state, batch_l, t)
    local u = {[t] = {}}
    for i = 1, #state do
      state[i]:zero()
      table.insert(u[t], state[i][{{1, batch_l}}])
    end
    if t == 0 then
      return u
    else
      return u[t]
    end      
  end

  function clean_layer(layer)
    if opt.gpuid >= 0 then
      layer.output = torch.CudaTensor()
      layer.gradInput = torch.CudaTensor()
    else
      layer.output = torch.DoubleTensor()
      layer.gradInput = torch.DoubleTensor()
    end
    if layer.modules then
      for i, mod in ipairs(layer.modules) do
        clean_layer(mod)
      end
    elseif torch.type(self) == "nn.gModule" then
      layer:apply(clean_layer)
    end      
  end

  -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
  function decay_lr(epoch)
    print(opt.val_perf)
    if epoch >= opt.start_decay_at then
      start_decay = 1
    end

    if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
      local curr_ppl = opt.val_perf[#opt.val_perf]
      local prev_ppl = opt.val_perf[#opt.val_perf-1]
      if curr_ppl > prev_ppl then
        start_decay = 1
      end
    end
    if start_decay == 1 then
      opt.learning_rate = opt.learning_rate * opt.lr_decay
    end
  end   

  function train_batch(data, epoch)
    local train_nonzeros = 0
    local train_loss = 0	       
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0


    for i = 1, data:size() do
      zero_table(grad_params, 'zero')
      local d
      if epoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]

      local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]
      local sent_enc_grads = sent_enc_grads[{{1, batch_l}, {1, source_l}}] -- added by Jeffrey

      local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
      local context = context_proto[{{1, batch_l}, {1, source_l}}]
      local sent_enc = sent_enc_proto[{{1, batch_l}, {1, source_l}}]
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
      end

      -- forward prop encoder
      for t = 1, source_l do
        encoder_clones[t]:training()
        local s = sent_conv_model:forward(source[t]) -- create s_t once
        sent_enc[{{},t}]:copy(s)
        local encoder_input = {s, table.unpack(rnn_state_enc[t-1])}
        local out = encoder_clones[t]:forward(encoder_input)
        rnn_state_enc[t] = out
        context[{{},t}]:copy(out[#out])
      end

      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid2)	    
        local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
        context2:copy(context)
        context = context2
      end

      -- forward prop decoder
      local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[0][L*2]:copy(rnn_state_enc[source_l][L*2-1])
          rnn_state_dec[0][L*2+1]:copy(rnn_state_enc[source_l][L*2])
        end
      end	 

      local preds = {} -- table of p_t
      local decoder_input
      for t = 1, target_l do
        decoder_clones[t]:training()
        local decoder_input = {sent_enc[t], context[t], table.unpack(rnn_state_dec[t-1])}
        local out = decoder_clones[t]:forward(decoder_input)
        local next_state = {}
        table.insert(preds, out[#out])
        table.insert(next_state, out[#out])
        for j = 1, #out-1 do
          table.insert(next_state, out[j])
        end
        rnn_state_dec[t] = next_state
      end

      -- backward prop decoder
      encoder_grads:zero()	 
      sent_enc_grads:zero()
      local drnn_state_dec = reset_state(init_bwd_dec, batch_l, 1)
      local loss = 0
      for t = target_l, 1, -1 do
        local pred = preds[t] -- batch_l x 2
        loss = loss + criterion:forward(pred, target_out[t])/batch_l
        local dl_dpred = criterion:backward(pred, target_out[t])
        dl_dpred:div(batch_l)
        drnn_state_dec[#drnn_state_dec]:add(dl_dpred)
        local decoder_input = {sent_enc[t], context[t], table.unpack(rnn_state_dec[t-1])}
        local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
        -- accumulate encoder/decoder grads
        encoder_grads[{{},t}]:add(dlst[2])
        drnn_state_dec[#drnn_state_dec]:zero()
        drnn_state_dec[#drnn_state_dec]:add(dlst[3]) -- backprop p_t
        for j = 4, #dlst do
          drnn_state_dec[j-3]:copy(dlst[j])
        end	    

        -- accumulate for conv
        sent_enc_grads[{{},t}]:add(dlst[1])
      end
      --word_vecs_dec.gradWeight[1]:zero()
      --if opt.fix_word_vecs_dec == 1 then
        --word_vecs_dec.gradWeight:zero()
      --end

      local grad_norm = 0
      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

      -- backward prop encoder
      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
        cutorch.setDevice(opt.gpuid)
        local encoder_grads2 = encoder_grad_proto2[{{1, batch_l}, {1, source_l}}]
        encoder_grads2:zero()
        encoder_grads2:copy(encoder_grads)
        encoder_grads = encoder_grads2 -- batch_l x source_l x rnn_size
      end

      local drnn_state_enc = reset_state(init_bwd_enc, batch_l, 1)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
        end	    
      end

      for t = source_l, 1, -1 do
        local encoder_input = {sent_enc[t], table.unpack(rnn_state_enc[t-1])}
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end	    

        -- accumulate for conv
        sent_enc_grads[{{},t}]:add(dlst[1])
      end

      -- backprop sent_conv
      for t = source_l, 1, -1 do
        sent_conv_model:forward(source[t]) -- inefficient, but needed if not cloning
        local dl_ds = sent_enc_grads[{{},t}]
        sent_conv_model:backward(source[t], dl_ds)
      end
      -- fix word embeddings
      word_vecs_enc.gradWeight[1]:zero()
      if opt.fix_word_vecs_enc == 1 then
        word_vecs_enc.gradWeight:zero()
      end

      grad_norm = (grad_norm + grad_params[1]:norm()^2)^0.5

      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_params do
        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
          if j == 1 then
            cutorch.setDevice(opt.gpuid)
          else
            cutorch.setDevice(opt.gpuid2)
          end
        end
        if shrinkage < 1 then
          grad_params[j]:mul(shrinkage)
        end	    
        params[j]:add(grad_params[j]:mul(-opt.learning_rate))
        param_norm = param_norm + params[j]:norm()^2
      end	    
      param_norm = param_norm^0.5

      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_nonzeros = train_nonzeros + nonzeros
      train_loss = train_loss + loss*batch_l
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
        local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
        epoch, i, data:size(), batch_l, opt.learning_rate)
        stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
        math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
        (num_words_target+num_words_source) / time_taken,
        num_words_source / time_taken,
        num_words_target / time_taken)			   
        print(stats)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end
    return train_loss, train_nonzeros
  end   

  local total_loss, total_nonzeros, batch_loss, batch_nonzeros
  for epoch = opt.start_epoch, opt.epochs do
    --generator:training()
    sent_conv_model:training()
    if opt.num_shards > 0 then
      total_loss = 0
      total_nonzeros = 0	 
      local shard_order = torch.randperm(opt.num_shards)
      for s = 1, opt.num_shards do
        local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
        print('loading shard #' .. shard_order[s])
        local shard_data = data.new(opt, fn)
        batch_loss, batch_nonzeros = train_batch(shard_data, epoch)
        total_loss = total_loss + batch_loss
        total_nonzeros = total_nonzeros + batch_nonzeros
      end
    else
      total_loss, total_nonzeros = train_batch(train_data, epoch)
    end
    local train_score = math.exp(total_loss/total_nonzeros)
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    opt.val_perf[#opt.val_perf + 1] = score
    decay_lr(epoch)
    -- clean and save models
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)      
    if epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(encoder); clean_layer(decoder); clean_layer(sent_conv_model) --clean_layer(generator)
      --torch.save(savefile, {{encoder, decoder, generator}, opt})
      torch.save(savefile, {{encoder, decoder, sent_conv_model}, opt})
    end
  end
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  clean_layer(encoder); clean_layer(decoder); clean_layer(sent_conv_model) -- clean_layer(generator)
  print('saving final model to ' .. savefile)   
  --torch.save(savefile, {{encoder:double(), decoder:double(), generator:double()}, opt})   
  torch.save(savefile, {{encoder:double(), decoder:double(), sent_conv_model:double()}, opt})   
end

function eval(data)
  encoder_clones[1]:evaluate()   
  decoder_clones[1]:evaluate() -- just need one clone
  sent_conv_model:evaluate()
  --generator:evaluate()
  local nll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 1)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    local sent_enc = sent_enc_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      local s = sent_conv_model:forward(source[t])
      sent_enc[{{},t}]:copy(s)
      local encoder_input = {s, table.unpack(rnn_state_enc)}
      local out = encoder_clones[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
    end

    if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2)
      local context2 = context_proto2[{{1, batch_l}, {1, source_l}}]
      context2:copy(context)
      context = context2
    end

    local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 1)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[L*2]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+1]:copy(rnn_state_enc[L*2])
      end	 
    end      
    local loss = 0
    for t = 1, target_l do
      local decoder_input = {sent_enc[t], context[t], table.unpack(rnn_state_dec)}
      local out = decoder_clones[1]:forward(decoder_input)
      rnn_state_dec = {}
      table.insert(rnn_state_dec, out[#out])
      for j = 1, #out-1 do
        table.insert(rnn_state_dec, out[j])
      end
      --local pred = generator:forward(out[#out])
      local pred = out[#out]
      loss = loss + criterion:forward(pred, target_out[t])
    end
    nll = nll + loss
    total = total + nonzeros
  end
  --local valid = math.exp(nll / total)
  local valid = nll / total -- loss (binary class) instead of perp
  print("Valid", valid)
  return valid
end


function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer	 
    elseif layer.name == 'word_vecs_enc' then
      word_vecs_enc = layer
    elseif layer.name == 'word_vecs' then
      word_vecs_dec = layer
      word_vecs_enc = layer
    elseif layer.name == 'decoder_attn' then	 
      decoder_attn = layer
    end
  end
end

function main() 
  -- parse input params
  opt = cmd:parse(arg)
  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    if opt.gpuid2 >= 0 then
      print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
    end      
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require 'cudnn'
    end      
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)      
  end

  -- Create the data loader class.
  print('loading data...')
  if opt.num_shards == 0 then
    train_data = data.new(opt, opt.data_file)
  else
    train_data = opt.data_file
  end

  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
  valid_data.source_size, valid_data.target_size))   
  opt.max_sent_l = math.max(valid_data.source:size(2), valid_data.target:size(2))
  if opt.use_chars_enc == 1 then
    -- MUST be used for document summary
    opt.max_word_l = valid_data.char_length
  end
  print(string.format('Source max sent len: %d, Target max sent len: %d',
  valid_data.source:size(2), valid_data.target:size(2)))   

  -- Build model
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc')
    decoder = make_lstm(valid_data, opt, 'dec')
    sent_conv_model = make_sent_conv(valid_data, opt) -- added by Jeffrey
    criterion = make_criterion(opt)
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    encoder = model[1]:double()
    decoder = model[2]:double()      
    --generator = model[3]:double()
    criterion = make_criterion(opt)
  end   

  layers = {encoder, decoder, sent_conv_model}

  if opt.gpuid >= 0 then
    for i = 1, #layers do	 
      if opt.gpuid2 >= 0 then 
        if i == 1 then
          cutorch.setDevice(opt.gpuid) --encoder on gpu1
        else
          cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
        end
      end	 
      layers[i]:cuda()
    end
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid2) --criterion on gpu2
    end      
    criterion:cuda()      
  end

  encoder:apply(get_layer)   
  decoder:apply(get_layer)   
  sent_conv_model:apply(get_layer)

  train(train_data, valid_data)
end

main()
