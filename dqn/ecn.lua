require 'nn'

local ecn, parent = torch.class('dqn.ecn', 'nn.Module')

function ecn:__init(args)
   print("Initializing ecn with args: ")
   for k,v in pairs(args) do
      print(k, v)
   end
   self.state_dim  = args.state_dim -- State dimensionality.
   self.actions    = args.actions
   self.n_actions  = #self.actions
   args.n_actions   = self.n_actions
   self.verbose    = args.verbose
   self.best       = args.best
   
   --- epsilon annealing
   self.ep_start   = args.ep or 1
   self.ep         = self.ep_start -- Exploration probability.
   self.ep_end     = args.ep_end or self.ep
   self.ep_endt    = args.ep_endt or 1000000

   ---- learning rate annealing
   self.lr_start       = args.lr or 0.01 --Learning rate.
   self.lr             = self.lr_start
   self.lr_end         = args.lr_end or self.lr
   self.lr_endt        = args.lr_endt or 1000000
   self.wc             = args.wc or 0  -- L2 weight cost.
   self.minibatch_size = args.minibatch_size or 1
   self.valid_size     = args.valid_size or 500

   --- Q-learning parameters
   self.discount       = args.discount or 0.99 --Discount factor.
   self.update_freq    = args.update_freq or 1
   -- Number of points to replay per learning step.
   self.n_replay       = args.n_replay or 1
   -- Number of steps after which learning starts.
   self.learn_start    = args.learn_start or 0
   -- Size of the transition table.
   self.replay_memory  = args.replay_memory or 1000000
   self.hist_len       = args.hist_len or 1
   self.rescale_r      = args.rescale_r
   self.max_reward     = args.max_reward
   self.min_reward     = args.min_reward
   self.clip_delta     = args.clip_delta
   self.target_q       = args.target_q
   self.bestq          = 0
   self.gpu            = args.gpu
   self.self_start     = args.self_start
   self.self_period    = args.self_period
   self.self_interval  = args.self_interval
   self.self_training  = args.self_trainingfalse

   if (self.gpu >= 0) then 
      require 'cunn'
      require 'cutorch'
   end


   self.ncols          = args.ncols or 1  -- number of color channels in input
   self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
   self.preproc        = args.preproc  -- name of preprocessing network
   self.histType       = args.histType or "linear"  -- history type to use
   self.histSpacing    = args.histSpacing or 1
   self.nonTermProb    = args.nonTermProb or 1
   self.bufferSize     = args.bufferSize or 512

   args.n_hid          = {512}

   self.eta = args.eta or 1
   
   self.transition_params = args.transition_params or {}
   self.ru = args.ru or nn.Reshape


   -- The convolutional network will take a state s of size (hist_len
   -- * screen_w * screen_h) as input and output a representation of
   -- that state.  The prediction network will take in the last frame
   -- fo the state (size screen_w * screen_h) and a vector
   -- representation of an action (size num_actions) and ouptut a
   -- prediction of the value of the representation of the
   -- convoultional network when run with input s2 (where if s
   -- represents frames <k, k+1, ..., k + hist_len>, s2 represents
   -- frames <k+1, k+2, ..., k + hist_len + 1>.

   -- Thus the input will consist of a minibatch of tensors of size
   -- w*h*hist_len + num_actions (we will use the last frame of the
   -- input state for both networks.  The network itself will consist
   -- of the convolutional network, the prediction network, a decision
   -- network (that outputs an estimated Q-value based on the output
   -- of the convolutional network) and some scaffolding (calls to
   -- narrow the input approriately to feed into the correct networks,
   -- concatenate outputs, etc.)

   
   self.input_size = (args.hist_len*args.input_dims[2] * args.input_dims[3]) + self.n_actions
   print("ECN input size is ", self.input_size)

   -- The network will be the main_network followed by the decnet.
   -- The main network will narrow the input appropriately and feed it
   -- into the convnet and prednet.
   self.main_network = nn.Concat(2) -- 1st dim is minibatch, concatenate along second dim
   self.convnet, convnet_num_features = self:create_convnet(args)
   self.prednet = self:create_prednet(args)
   
   cbranch = nn.Sequential()
   self.conv_narrow = nn.Narrow(2, 1, args.hist_len*args.input_dims[2]*args.input_dims[3])
   cbranch:add(self.conv_narrow)
   cbranch:add(self.convnet)
   self.main_network:add(cbranch)
   self.convout_dims = {8, 7, 7}  --This should be computed; hardcoding for now.

   self.pbranch = nn.Sequential()
   self.pred_narrow = nn.Narrow(2, (args.hist_len -1)*args.input_dims[2]*args.input_dims[3] + 1, ( args.input_dims[2] * args.input_dims[3] + self.n_actions))
   self.pbranch:add(self.pred_narrow)
   self.pbranch:add(self.prednet)
   self.main_network:add(self.pbranch)

   
   self.network = nn.Sequential()
   self.network:add(self.main_network)

   -- We only want to feed the output of the convnet into the decnet
   -- We want a narrowed decnet (the last two components of the network) to use for backward passes

   self.narrow_decnet = nn.Sequential()
   self.narrow_decnet:add(nn.Narrow(2, 1, convnet_num_features))
   args.convout_nel = (convnet_num_features) * 7 * 7 -- Hardcoding in 7 for now!
   self.decnet = self:create_mlp(args)
   self.narrow_decnet:add(self.decnet)
   self.network:add(self.narrow_decnet)






   --pred_net:add(nn.Reshape(7,7))  -- maybe this sohuld go in the next part; nice to keep these as 1d tensors
		

   -- The main network will consist of the convnet and pred_net
   --local main_network = nn.Concat(2)
   --main_network:add(self.convnet)
   --main_network:add(self.pred_net)
   -- A Concat network produces an output of size |out_{conv}| +
   -- |out_{pred}|.  We only really want the output from the
   -- convolutional network to keep on.
   
   --self.network = nn.Sequential()
   --self.network:add(main_network)

   --Compute the size of the output of the convnet and narrow the
   --output of the entire network.  There's probably a better way to
   --do this, but for now we'll just create fake input, run it through
   --the network, and get the output sizes.
   print("Here is the main network:")
   print(self.main_network)


   -- print("Input size is " .. self.input_size)
   -- print("First narrow call starts at  1 with len " .. args.hist_len*args.input_dims[2]*args.input_dims[3])
   -- print("Second narrow call start at " .. penultimate_screen_offset .. " with length " .. pred_vector_len)
   -- z = torch.zeros(32, self.input_size)
   -- self.convnet:forward(z)
   -- convnet_num_features = self.convnet.output:size(2)
   -- print("Made it out of convnet.")
   -- print("Output size is " .. tostring(self.convnet.output:size()))
   -- print("Num features is " .. tostring(convnet_num_features))
   -- args.convout_nel = convnet_num_features * 7 * 7 -- Hardcoding in 7 for now!
   -- self.pred_net:forward(z)
   -- print("Made it out of pred_net.")
   -- print("Output size is " .. tostring(self.pred_net.output:size()))
   -- main_network:forward(torch.zeros(32,self.input_size))
   -- print("main output size is " .. tostring(main_network.output:size()))
   -- conv_outsize = self.convnet.output:size(2)
   -- pred_outsize = self.pred_net.output:size(2)
   -- self.network:add(nn.Narrow(2, 1, convnet_num_features))
   
   -- Add the final layers and create a simplified (non-enactive)
   -- version of the network; this will be used to compute best moves
   -- (not in learning).  The simplified network is basically the
   -- convolutional_network followed by the decision network.  The
   -- conv_net has an initial Narrow that we'll get rid of, since it's
   -- expecting tensors which have the action info as well and
   -- narrowing them into just the screen info.

   
   --self. = self:create_mlp(args)
   --self.network:add(self.mlp)
   self.simple_network = nn.Sequential()
   self.simple_network:add(self.conv_narrow)
   self.simple_network:add(self.convnet)   
   self.simple_network:add(self.decnet)

   -- Predictive training network will ignore the decnet and make sure
   -- that only the last screen is fed into the convnet.  The latter
   -- will consist of narrowing the input and the padding with zeros
   -- to give the effect of having only 0s in the first three screens.

   self.pred_training_network = nn.Concat(2)
   -- The predictor for the pred_training_network will just be pbranch from before

   
   -- The hierarchical feature extractor for the pred_training_network
   self.pred_training_hier_network = nn.Sequential()
   local frame_size = self.input_dims[2]*self.input_dims[3]
   -- Pick out the last frame
   self.pred_training_narrow =  nn.Narrow(2, (self.hist_len - 1)*frame_size + 1, frame_size)
   -- Pad in zero for the frames before zero
   self.pred_training_padding = nn.Padding(2, -(self.hist_len - 1)*frame_size)
   self.pred_training_hier_network:add(self.conv_narrow)
   self.pred_training_hier_network:add(self.pred_training_narrow)
   self.pred_training_hier_network:add(self.pred_training_padding)   
   self.pred_training_hier_network:add(self.convnet)
   self.pred_training_hier_network:add(nn.Select(2,1))
   self.pred_training_hier_network:add(self.ru(1,7,7))
   --self.pred_training_hier_network:add(nn.Reshape(1,7,7))

   self.pred_training_network:add(self.pbranch)
   self.pred_training_network:add(self.pred_training_hier_network)

   print("The pred_training_network")
   print(self.pred_training_network)

   -- Sanity checks
   if (args.gpu >= 0) then
      self.pred_training_network:cuda()
      self.convnet:cuda()
      self.prednet:cuda()
      self.decnet:cuda()
      self.main_network:cuda()
      self.simple_network:cuda()
      self.network:cuda()
      z1 = torch.Tensor(self.minibatch_size, self.input_size):contiguous():cuda()
      z2 = self.pred_training_network:forward(z1)
      z2:contiguous():cuda()
      
      print("Output size from pred_training_network is ", z2:size())
      self.pred_training_network:backward(z1:contiguous(), z2:contiguous())
      local dC = z2:select(2,1):reshape(self.minibatch_size, 1, 7, 7):cuda()
      local dP = z2:select(2,2):reshape(self.minibatch_size, 1, 7, 7):cuda()
      print("dC, dP sizes are ", dC:size(), dP:size())
      print("Hier netwwork", self.pred_training_hier_network)
      print("Hier network output size is ", self.pred_training_hier_network.output:size())
      self.pred_training_hier_network:updateGradInput(z1, dC)
      self.pbranch:updateGradInput(z1, dP)
      print("Went backward through pred_training_network successfully.")
   end
   
   print("The final network")
   print(self.network)
   if args.gpu >= 0 then
      require 'cudnn'
      self.convnet:cuda()
      self.prednet:cuda()
      self.decnet:cuda()
      self.main_network:cuda()
      self.simple_network:cuda()
      self.network:cuda()
      self.pred_training_network:cuda()
      cudnn.convert(self.network)
      print("CUDAfied network")
   else
      self.network:float()
      print("Floatified network")
   end

   print("Testing forward pass.")
   -- --self.network:float()
   z1 = torch.Tensor(self.minibatch_size, self.input_size)
   if (args.gpu >= 0) then
      z1 = z1:cuda()
   else
      z1:float()
   end
   z2 = self.network:forward(z1)
   print("Forward output size is ", z2:size())
   

   print("Testing backward pass.")
   grad = self.network:backward(z1, z2)
   
   -- z2 = self.pred_net:forward(z1)
   -- self.pred_net:updateGradInput(z1, z2)
   -- print("Done with pred_net.  i/o sizes are " .. tostring(z1:size()) .. '/' .. tostring(z2:size()))
   -- z2 = self.convnet:forward(z1)
   -- self.convnet:updateGradInput(z1, z2)
   -- print("Done with convnet.  i/o sizes are " .. tostring(z1:size()) .. '/' .. tostring(z2:size()))
   -- z2 = self.simple_network:forward(z1)
   -- self.simple_network:updateGradInput(z1, z2)
   -- print("Done with simple.  i/o sizes are " .. tostring(z1:size()) .. '/' .. tostring(z2:size()))
   -- z2 = self.network:forward(z1)
   -- self.network:updateGradInput(z1, z2)
   -- print("Done with full.  i/o sizes are " .. tostring(z1:size()) .. '/' .. tostring(z2:size()))
      
   -- The loss of the prediction network will be separate from the
   -- output of the network.
   self.predcost = nn.MSECriterion()
   self.predcost.sizeAverage = false

   
end

function ecn:conv_input()
   return self.conv_narrow.output
end

function ecn:pred_input()
   return self.pred_narrow.output
end

function ecn:create_convnet(args)
   args.n_units        = {32, 64, 8}
   args.filter_size    = {8, 4, 3}
   args.filter_stride  = {4, 2, 1}
   args.n_hid          = {512}
   args.nl             = nn.Rectifier
   
   local net = nn.Sequential()
   net:add(self.ru(unpack(self.input_dims)))
   --net:add(nn.Reshape(unpack(self.input_dims)))
   
   --- first convolutional layer
   local convLayer = nn.SpatialConvolution
   
   net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
		     args.filter_size[1], args.filter_size[1],
		     args.filter_stride[1], args.filter_stride[1],1))
   net:add(args.nl())
   
   -- Add convolutional layers
   for i=1,(#args.n_units-1) do
      -- second convolutional layer
      net:add(convLayer(args.n_units[i], args.n_units[i+1],
			args.filter_size[i+1], args.filter_size[i+1],
			args.filter_stride[i+1], args.filter_stride[i+1]))
      net:add(args.nl())
   end

   --The old return
   return net, args.n_units[table.getn(args.n_units)]
end

-- The predictor
function ecn:create_prednet(args)   
   pred_net = nn.Sequential()
   pred_input_len = ( args.input_dims[2] * args.input_dims[3] + args.n_actions )
   pred_net:add(nn.Linear(pred_input_len  , 512))
   pred_net:add(nn.Tanh())
   pred_net:add(nn.Linear(512, 49))
   pred_net:add(nn.Tanh())
   --pred_net:add(nn.Reshape(1,7,7))
   pred_net:add(self.ru(1,7,7))
   return pred_net
end


function ecn:create_mlp(args)
   local net = nn.Sequential()

   -- if args.gpu >= 0 then
   --    nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
   -- 			       :cuda()):nElement()
   -- else
   --    nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
   -- end

   -- reshape all feature planes into a vector per example
   --net:add(nn.Reshape(args.convout_nel))
   net:add(self.ru(args.convout_nel))
   
   -- fully connected layer
   net:add(nn.Linear(args.convout_nel, args.n_hid[1]))
   net:add(args.nl())
   local last_layer_size = args.n_hid[1]
   
   for i=1,(#args.n_hid-1) do
      -- add Linear layer
      last_layer_size = args.n_hid[i+1]
      net:add(nn.Linear(args.n_hid[i], last_layer_size))
      net:add(args.nl())
   end
   
   -- add the last fully connected layer (to actions)
   net:add(nn.Linear(last_layer_size, args.n_actions))
   
    if args.gpu >=0 then
       net:cuda()
    end
    if args.verbose >= 2 then
       print(net)
       print('Convolutional layers flattened output size:', args.convout_nel)
    end
    return net
end

-- Assuming an input has been passed through the network, get the minibatch of first output features from the convnet
function ecn:get_f1()
   return self.main_network.output:select(2,1)
end

-- Assuming an input has been passed through the network, get the minibatch of first output features from the convnet
function ecn:save_f1(f1_val)
   self.f1 = f1_val:clone()
end


-- Our total loss will be the loss from the decision network (D)
-- (determined in the encapsualating class, NeuralQLearner right now)
-- plus the loss from the enactive part (E).  Thus the gradient at I
-- is dD/dI + dE/DI.
--
-- If we let J denote the output of the main_network (== the input to
-- the decision network), let C be the convnet and let P be the
-- prednet, then we have:
--  dL/dI = dD/dI + dE/dI
--        = dD/dJ dJ/dI + dE/dJ dJ/dI
--        = (dD/dJ + dE/dJ) dJ/dI
--
-- We can calculate dD/dJ by running gradOutput backward through the
-- decision + narrowing network.  We will calculate dE/dJ here, then
-- we run (dD/dJ + dE/dJ) backward through the main network.
--
-- (Eventually, the decision network should probably be moved into
-- NeuralQLearner to maintin the proper conceptual boundaries).

function ecn:updateGradInput(input, gradOutput)
   --print("Updating grad Input (size ", input:size(), " with gradOutput of size ", gradOutput:size())
   local J = self.main_network.output
   local dD = self.narrow_decnet:updateGradInput(J, gradOutput)


   --The weights that are updated are those of the predictive network
   --and the conv_net.  The update is weighted by the parameter
   --self.eta.

   --The predictive network is updated by standard backprop using the
   --mean-squared error between the first feature and the predicted
   --feature.

   --The convnet is updated in the same way

   --assert(self.f1)
   --print("f1 size is ", self.f1:size())
   prediction = self.prednet.output:clone()
   --print("prediction size is ", prediction:size())
   dC = (self.f1 - prediction):mul(self.eta)
   dP = dC:clone():mul(-1)

   -- The output J (and thus dD) of the main_network will be a tensor
   -- of size minibatch_size x (num_conv_features + 1 + 1) x 7 x 7 (at
   -- least for the current config).

   -- The first num_conv_features in dimension 2 correspond to the
   -- conv_net, the penultimate feature of the output is f1, and the
   -- last is the prediction.
   -- JB:  This has changed from before.
   dD:select(2,9):add(dC)
   dD:select(2,10):add(dP)   
   self.main_network:updateGradInput(input, dD)
   return self.gradInput
end


-- function ecn:updateGradInput(input, gradOutput)
--    -- The gradient will be computed based on the error,
--    -- which in this case will the MSE between the predicted
--    -- value of the first feature (in the last layer of the convent)
--    -- and the actual value (which is the output for the entire network)
--    --
--    -- I'm *still* a little confused about exactly what this is supposed to compute.
--    -- For now, I'll return dLoss/dInput.  I want the total loss to be given by
--    -- L = MSE(prediction, first_feature) + loss_{convnet}
--    -- Thus dL/dI = M' + l' = (dM/dP)(dP/dI) + (dM/dF)(dF/dI) + dC/dI
--    -- The last I think gets handled in NeuralQLearner code; although
--    -- this may be precisely where gradOutput comes in.  I think it is!
--    -- Then we should have dC/dI = (dC/dO)(dO/dI) where dC/dO = gradOutput
--    -- and dO/dI is the gradient of the convnet
--    --
--    -- For me, here's what updateGradInput does.  Let E(I) = ecn(I) and
--    -- suppose we want to get the gradient of F(E(I)) wrt I.
--    -- Then we (dF/dE)(dE/dI):  dF/dE comes in in gradOutput, it's our job to
--    -- compute dE/dI.

--    -- Pick out the three main subnetworks
--    main_network = self.network:get(1)
--    narrow_network = self.network:get(2)
--    decision_network = self.network:get(3)

--    -- Get the feature and the prediction and calculate the MSE between them.
   
--    feature = main_network:get(1).output:select(2,1)
--    prediction = main_network:get(2).output:select(2,1)
--    msep = self.predcost:forward(prediction, feature)
--    local mse_wrtp_grad = self.predcost:updateGradInput(prediction, feature)  -- This is probaby not as symmetric as it looks!
--    msef = self.predcost:forward(feature, prediction)
--    local mse_wrtf_grad = self.predcost:updateGradInput(feature, prediction)  -- So we get the other way too



--    main_output = main_network.output
--    narrow_output = narrow_network.output
--    decision_gip = decision_network:updateGradInput(narrow_output, gradOutput)
--    narrow_gip = narrow_network:updateGradInput(main_output, decision_gip)

--    -- Just for testing
--    main_network:updateGradInput(input, narrow_gip)

--    --print("Size of narrow_gip is: " .. tostring(narrow_gop:size()))
--    --print("Sizes of mse_wrtp_grad, msef_grad are: "
-- 	 --   .. tostring(mse_wrtp_grad:size()) .. tostring(mse_wrtf_grad:size()))


--    -- narrow_gop is batchsize x 65 x 7 x 7.
--    -- In each batch, the first 64 tensors are for the conv_net
--    -- while the last one is for the prednet.
--    -- We want to add mse_wrtp_grad to the prednet and mse_wrtf_grade
--    -- to the first feature of the convnet.  We'll use select.
--    predg = narrow_gip:select(2,65)
--    convfg = narrow_gip:select(2,1)
--    predg = predg + mse_wrtp_grad
--    convfg = convfg + mse_wrtf_grad

--    main_network:updateGradInput(input, narrow_gip)
--    return self.gradInput
-- end

-- Don't forget to add the mask for the pred network.
function ecn:updateOutput(input)
   if (self.gpu >=0) then
      input = input:contiguous():cuda()
   end
   return self.network:updateOutput(input)
end

-- Given an input, compute the gradient of the loss for the predictive network
function ecn:pred_training_updateGradients(input)

   --The weights that are updated  are those of the predictive network and the conv_net.  The update is weighted by the parameter self.eta.
   --The predictive network is updated by standard backprop using the mean-squared error between the first feature and the predicted feature.
   --The convnet is updated in the same way

   prediction = self.prednet.output
   
   --print("prediction size is ", prediction:size())

   dC = (self.f1 - prediction):mul(self.eta)
   dP = dC:clone():mul(-1)

   -- -- Recall that the pred_training_network is the predictive network followed by the feature.
   -- dD = torch.cat(dP, dC, 1)
   in_shape = dC:size()
   dC = dC:reshape(in_shape[1], 1, in_shape[2], in_shape[3])
   dP = dP:reshape(in_shape[1], 1, in_shape[2], in_shape[3])
   -- print("dD size is ", dD:size())
   self.pbranch:updateGradInput(input, dP)
   self.pred_training_hier_network:updateGradInput(input, dC)
   self.pbranch:updateGradInput(input, dP)
   return self.pred_training_network.gradInput
end


function ecn:dqn_updateGradInput(input, gradOutput)
   --debugging
   -- w, dw = self.simple_network:getParameters()
   -- print("Sum of weights, dw, input, gradOutput before update are" .. w:sum() .. " " .. dw:sum() .. " " .. input:sum() .. " " .. gradOutput:sum())
   -- v = self.simple_network:updateGradInput(input, gradOutput)
   -- print("Sum of weights after update is" .. w:sum()  .. " " .. dw:sum().. " " .. input:sum() .. " " .. gradOutput:sum())
   -- return v
   -- regular function; return to this when finished debugging.
   return self.simple_network:updateGradInput(input, gradOutput)
end

function ecn:dqn_accGradParameters(input, gradOutput, scale)
   return self.simple_network:accGradParameters(input, gradOutput, scale)
end

function ecn:parameters()
   return self.network:parameters()
end

-- Pass an input through the network as previously defined (without the enactive predictor)
function ecn:simple_forward(input)
   return self.simple_network:forward(input)
end
