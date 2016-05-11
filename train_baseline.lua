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
cmd:option('-savefile', 'doc_summary', [[Savefile name (model will be saved as 
savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is 
the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards, 
then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
pretrained model.]])

cmd:option('-test_only', 0, [[Test only. Requires predfile and train_from options set]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-word_vec_size', 300, [[Word embedding sizes]]) -- 300 for word2vec
cmd:option('-use_chars_enc', 1, [[If 1, use character on the encoder 
side (instead of word embeddings - this is now required for the document encoder]])

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
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-pre_word_vecs_enc', 'data/word2vec.hdf5', [[If a valid path is specified, then this will load 
pretrained word embeddings (hdf5 file) on the encoder side. 
See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")


cmd:option('-start_symbol', 1, [[Use special start-of-sentence and end-of-sentence tokens
on the source side. We've found this to make minimal difference]])
cmd:option('-predfile', '', [[File to write predictions to, empty for no predictions]])
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

function train(train_data, valid_data)

  local timer = torch.Timer()
  local num_params = 0
  local start_decay = 0
  opt.train_perf = {}
  opt.val_perf = {}

  local p, gp = logreg:getParameters()
  if opt.train_from:len() == 0 then
    p:uniform(-opt.param_init, opt.param_init)
  end
  num_params = num_params + p:size(1)
  params = p
  grad_params = gp

  if opt.pre_word_vecs_enc:len() > 0 then   
    print('Loading pretrained embeddings')
    local f = hdf5.open(opt.pre_word_vecs_enc)     
    local pre_word_vecs = f:read('word_vecs'):all()
    for i = 1, pre_word_vecs:size(1) do
      word_vecs_enc.weight[i]:copy(pre_word_vecs[i])
    end      
  end

  print("Number of parameters: " .. num_params)

  -- clone up to max source/target length
  logreg_clones = clone_many_times(logreg, opt.max_sent_l)

  word_vecs_enc.weight[1]:zero()            

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
      grad_params:zero()
      local d
      if epoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]
      assert(source_l == target_l, source_l .. " " .. target_l)


      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
      end

      -- forward 
      local preds = {}
      for t = 1, source_l do
        logreg_clones[t]:training()
        local pred = logreg_clones[t]:forward(source[t])
        table.insert(preds, pred)
      end

      -- backward 
      local loss = 0
      for t = 1, target_l do
        loss = loss + criterion:forward(preds, target_out[t])/batch_l
        local dl_dpred = criterion:backward(pred, target_out[t])
        dl_dpred:div(batch_l)
        logreg_clones[t]:backward(source[t], dl_dpred)
      end

      -- fix word embeddings
      word_vecs_enc.gradWeight[1]:zero()
      if opt.fix_word_vecs_enc == 1 then
        word_vecs_enc.gradWeight:zero()
      end

      local grad_norm = grad_params:norm()

      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      if shrinkage < 1 then
        grad_params:mul(shrinkage)
      end	    
      params:add(grad_params:mul(-opt.learning_rate))
      param_norm = params[j]:norm()

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
        math.exp(train_loss/train_nonzeros), param_norm, grad_norm) -- TODO: check what nonzeros is
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
    clean_layer(logreg)
    if epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      torch.save(savefile, {logreg, opt})
    end
  end

  if opt.predfile:len() > 0 then   
    print('Generating and writing predictions...')
    eval(valid_data, 1) -- 1 for predict
  end
  -- save final model
  local savefile = string.format('%s_final.t7', opt.savefile)
  clean_layer(logreg)

  print('saving final model to ' .. savefile)
  torch.save(savefile, {logreg:double(), opt})
end


function eval(data, do_predict)
  do_predict = do_predict or 0

  logreg_clones[1]:evaluate()

  -- for predict
  local predictions
  if do_predict == 1 then
    predictions = torch.zeros(data.target:size())
  end
  local pred_cur = 1

  local nll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]

    -- forward
    local preds = {}
    for t = 1, source_l do
      local pred = logreg_clones[1]:forward(source[t])
      table.insert(preds,pred)
    end

    local loss = 0
    for t = 1, target_l do
      local pred = preds[t]
      loss = loss + criterion:forward(pred, target_out[t])
      if do_predict == 1 then
        predictions:sub(pred_cur, pred_cur+batch_l-1,t,t):copy(pred)
      end
    end
    pred_cur = pred_cur + batch_l
    nll = nll + loss
    total = total + nonzeros
  end
  local valid = math.exp(nll / total)
  print("Valid", valid)

  if do_predict == 1 then
    local f = hdf5.open(opt.predfile, 'w')
    f:write('preds', predictions)  
    f:close()
  end
  return valid
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs' then
      -- we don't have decoder word vecs.
      word_vecs_enc = layer
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
    print('Building model')
    logreg = make_baseline(valid_data, opt)
    criterion = make_criterion(opt)
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    logreg = model[1]:double()
    criterion = make_criterion(opt)
  end   

  if opt.gpuid >= 0 then
    logreg = logreg:cuda()
    criterion:cuda()      
  end

  logreg:apply(get_layer)   

  if opt.test_only == 1 then
    print('Test only')
    assert(opt.predfile:len() > 0, 'predfile needed')
    assert(path.exists(opt.train_from), 'train_from needed')

    -- Do some annoying inits
    print('Generating and writing predictions...')
    eval(valid_data, 1) -- 1 for predict
  else
    -- Train as usual
    print('Training...')
    train(train_data, valid_data)
  end
end

main()
