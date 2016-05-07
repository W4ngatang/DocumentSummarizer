function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function make_sent_conv(data, opt)
  local input_size = opt.num_kernels

  local input = nn.Identity()()
  local word_vecs = nn.LookupTable(data.char_size, opt.word_vec_size)
  word_vecs.name = 'word_vecs'
  local wordcnn = make_cnn(opt.word_vec_size,  opt.kernel_width, opt.num_kernels)
  wordcnn.name = 'wordcnn'
  x = wordcnn(word_vecs(input))
  if opt.num_highway_layers > 0 then
    local mlp = make_highway(input_size, opt.num_highway_layers)
    mlp.name = 'mlp'
    x = mlp(x)
  end
  return nn.gModule({input}, {x})
end

-- Encoder: sentence LSTM
-- Decoder: sentence LSTM with encoded state h_t
function make_lstm(data, opt, model)
   assert(model == 'enc' or model == 'dec')
   --local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local input_size = opt.num_kernels -- size after conv

   local offset = 0
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   if model == 'dec' then
      -- TODO: this needs to be bidirectional
      table.insert(inputs, nn.Identity()()) -- source context h_t (batch_size x rnn_size)
      table.insert(inputs, nn.Identity()()) -- previous prob p_{t-1} (batch_size x 2)
      offset = offset + 2
   end   
   for L = 1,n do
     table.insert(inputs, nn.Identity()()) -- prev_c[L]
     table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local enc_sent
   local outputs = {}
   for L = 1,n do
     -- c,h from previous timesteps
     local prev_c = inputs[L*2+offset]    
     local prev_h = inputs[L*2+1+offset]
     -- the input to this layer
     if L == 1 then
       if model == 'enc' then
         -- encoder
         x = inputs[1]
         input_size_L = input_size
       else
         -- decoder
         x = inputs[1]
         local prob = nn.Select(2,2)(inputs[offset+1]) -- batch_size x 1
         prob = nn.Replicate(input_size,2)(prob) -- batch_size x input_size
         x = nn.CMulTable()({prob, x})
         input_size_L = input_size
       end
     else
       x = outputs[(L-1)*2]
       -- TODO: removed a res net
       input_size_L = rnn_size
       -- TODO: removed a hop attn here, consider adding it
       if dropout > 0 then
         x = nn.Dropout(dropout, nil, true)(x)
       end
     end
     -- evaluate the input sums at once for efficiency
     local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
     local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
     local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  if model == 'dec' then
    local top_h = outputs[#outputs]
    local term1 = nn.LinearNoBias(rnn_size, 2)(top_h)
    local term2 = nn.LinearNoBias(rnn_size, 2)(inputs[offset])
    local pred_prob = nn.LogSoftMax()(nn.CAddTable()({term1, term2}))
    table.insert(outputs, pred_prob) -- p_t for sentence label, batch_l x 2
  end
  return nn.gModule(inputs, outputs)
end

function make_criterion(opt)
   local w = torch.ones(2)
   criterion = nn.ClassNLLCriterion(w) -- TODO: ???
   criterion.sizeAverage = false
   return criterion
end

-- annoying hack for Sum backward prop
Sum_nc, _ = torch.class('nn.Sum_nc', 'nn.Sum')
function Sum_nc:updateGradInput(input, gradOutput)
    local size = input:size()
    size[self.dimension] = 1
    -- modified code:
    if gradOutput:isContiguous() then
        gradOutput = gradOutput:view(size) -- doesn't work with non-contiguous tensors
    else
        gradOutput = gradOutput:resize(size) -- slower because of memory reallocation and changes gradOutput
        -- gradOutput = gradOutput:clone():resize(size) -- doesn't change gradOutput; safer and even slower
    end
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput:expandAs(input))
    return self.gradInput
end 

-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
   local output
   local input = nn.Identity()() 
   if opt.cudnn == 1 then
      local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
					    kernel_width, 1, 1, 0)
      local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
      output = nn.Sum_nc(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
   else
      local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
      local conv_layer = conv(input)
      output = nn.Max(2)(nn.Tanh()(conv_layer))
   end
   return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local num_layers = num_layers or 1
    local input_size = input_size
    local output_size = output_size or input_size
    local bias = bias or -2
    local f = f or nn.ReLU()
    local start = nn.Identity()()
    local transform_gate, carry_gate, input, output
    for i = 1, num_layers do
       if i > 1 then
	  input_size = output_size
       else
	  input = start
       end       
       output = f(nn.Linear(input_size, output_size)(input))
       transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
					nn.Linear(input_size, output_size)(input)))
       carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
       local proj
       if input_size==output_size then
	  proj = nn.Identity()
       else
	  proj = nn.LinearNoBias(input_size, output_size)
       end
       input = nn.CAddTable()({
	                     nn.CMulTable()({transform_gate, output}),
                             nn.CMulTable()({carry_gate, proj(input)})})
    end
    return nn.gModule({start},{input})
end
