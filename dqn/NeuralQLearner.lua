--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
  require 'initenv'
end

require 'ecn'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
  self.state_dim  = args.state_dim -- State dimensionality.
  self.actions    = args.actions
  self.n_actions  = #self.actions
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
  --self.minibatch_size = args.minibatch_size or 1
  self.minibatch_size = args.minibatch_size or 64
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

  self.ncols          = args.ncols or 1  -- number of color channels in input
  self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
  self.preproc        = args.preproc  -- name of preprocessing network
  self.histType       = args.histType or "linear"  -- history type to use
  self.histSpacing    = args.histSpacing or 1
  self.nonTermProb    = args.nonTermProb or 1
  self.bufferSize     = args.bufferSize or 512

  self.transition_params = args.transition_params or {}
  self.eta            = args.eta or 0.1
  print("args.eta, self.eta are ", args.eta, self.eta)
  print("args.minibatch_size, self.minibatch_size are ", args.minibatch_size, self.minibatch_size)

  -- This isn't really the right place to set these, but want to move forward.  Fix this later.
  -- self_period:  how frequently do we try to learn self
  -- self_interval:  for how many steps do we try to learn self.
  self.self_period    = args.self_period or   200000
  self.self_interval  = args.self_interval or 2000000 -- change to something bigger; just debugging now.
  self.self_training  = args.self_training or false

  --debugging
  self.self_period    = args.self_period or 2000
  self.self_interval  = args.self_interval or 10000 -- change to something bigger; just debugging now.
  self.self_training  = args.self_training or false
  print("self_start, etc are: ", self.self_start, self.self_period, self.self_interval)

  -- self.enactiveLearning = false  -- Learn the DQN or the enactive part; DQN to start with.
  -- self.learningSwitch =  2000000000  -- How often to switch between learning enactive and DQN.


  --self.network    = args.network or self:createNetwork()

  -- check whether there is a network file
  --local network_function
  --if not (type(self.network) == 'string') then
  --        error("The type of the network provided in NeuralQLearner" ..
  --        " is not a string!")
  --end

  --local msg, err = pcall(require, self.network)
  -- if not msg then
  --     -- try to load saved agent
  --     local err_msg, exp = pcall(torch.load, self.network)
  --     if not err_msg then
  --         error("Could not find network file ")
  --     end
  --     if self.best and exp.best_model then
  --         self.network = exp.best_model
  --     else
  --         self.network = exp.model
  --     end
  -- else
  --     print('Creating Agent Network from ' .. self.network)
  --     self.network = err
  --     self.network = self:network()
  -- end
  args.input_dims = self.input_dims
  args.hist_len = self.hist_len
  self.network = dqn.ecn(args)
  if self.gpu and self.gpu >= 0 then
    self.network:cuda()
  else
    self.network:float()
  end

  print("Printing network.")
  print(self.network)

  -- Load preprocessing network.
  if not (type(self.preproc == 'string')) then
    error('The preprocessing is not a string')
  end
  msg, err = pcall(require, self.preproc)
  if not msg then
    error("Error loading preprocessing net")
  end
  self.preproc = err
  self.preproc = self:preproc()
  self.preproc:float()

  -- Load dqn
  psd = torch.DiskFile('outputs.psd/psd,encoderType=tanh,kernelsize=9,lambda=1/models/model.bin','r'):binary():readObject()
  print("Printing PSD encoder")
  print(psd.encoder)

  if self.gpu and self.gpu >= 0 then
    self.network:cuda()
    self.tensor_type = torch.CudaTensor
  else
    self.network:float()
    self.tensor_type = torch.FloatTensor
  end

  -- Create transition table.
  ---- assuming the transition table always gets floating point input
  ---- (Float or Cuda tensors) and always returns one of the two, as required
  ---- internally it always uses ByteTensors for states, scaling and
  ---- converting accordingly
  local transition_args = {
    stateDim = self.state_dim, numActions = self.n_actions,
    histLen = self.hist_len, gpu = self.gpu,
    maxSize = self.replay_memory, histType = self.histType,
    histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
    bufferSize = self.bufferSize
  }

  self.transitions = dqn.TransitionTable(transition_args)

  self.numSteps = 0 -- Number of perceived states.
  self.lastState = nil
  self.lastAction = nil
  self.v_avg = 0 -- V running average.
  self.tderr_avg = 0 -- TD error running average.

  self.q_max = 1
  self.r_max = 1

  self.w, self.dw = self.network:getParameters()

  self.simpw, self.simpdw = self.network.simple_network:getParameters()
  self.predw, self.pred_dw = self.network.prednet:getParameters()
  self.convw, self.conv_dw = self.network.convnet:getParameters()

  self.dw:zero()

  self.deltas = self.dw:clone():fill(0)

  self.tmp= self.dw:clone():fill(0)
  self.g  = self.dw:clone():fill(0)
  self.g2 = self.dw:clone():fill(0)

  if self.target_q then
    self.target_network = self.network:clone()
  end

  self.C = torch.Tensor(self.minibatch_size, unpack(self.network.convout_dims)):zero()
  if self.gpu and self.gpu >= 0 then
    self.C = self.C:cuda()
  else
    self.C = self.C:float()
  end

end


function nql:reset(state)
  print("In nql:reset")
  if not state then
    return
  end
  self.best_network = state.best_network
  self.network = state.model
  self.w, self.dw = self.network:getParameters()
  self.dw:zero()
  self.numSteps = 0
  self.next_f1 = nil

  print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
  if self.preproc then
    return self.preproc:forward(rawstate:float())
    :clone():reshape(self.state_dim)
  end
  -- Run the forward pass on the encoder here

  return rawstate
end


-- Take a tensor of actions in which action is represented as an
-- integer in the range 1 to n_actions and return a tensor in which
-- each action is represented as a vector of length n_actions which is
-- 1 at the index of the action being represented and 0 elsewhere

function nql:makeActionTensor(a)
  action_tensor = torch.Tensor(self.minibatch_size, self.n_actions)
  for j = 1, self.minibatch_size do
    action_tensor[j]:zero()
    action_tensor[j][a[j]] = 1
  end
  if self.gpu and self.gpu >= 0 then
    action_tensor = action_tensor:cuda()
  else
    action_tensor = action_tensor:float()
  end
  return action_tensor
end


-- Do a forward pass of the network.  We use minibatches s, a, r, s2
-- where for each index, the entry in s represents the state (that is
-- a concatenated vector of hist_len screen frames), the entry in a
-- represents the action taken after the last frame in s, and s2
-- represents the next state (the hist_len frames starting just after
-- the frames in s).  In particular, for training the predictive
-- network, we'll need the last frame from s, the action from a, and
-- the last frame from s2.  We'll save these in class-level variables
-- to be used in the next backward pass through the network.
--



function nql:getQUpdate(args)
  local s, a, r, s2, term, delta
  local q, q2, q2_max

  s = args.s
  a = args.a
  r = args.r
  s2 = args.s2
  term = args.term
  backprop = args.backprop  -- We'll use the simple forward if not doing backprop

  -- The order of calls to forward is a bit odd in order
  -- to avoid unnecessary calls (we only need 2).

  -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
  term = term:clone():float():mul(-1):add(1)

  local target_q_net
  if self.target_q then
    target_q_net = self.target_network
  else
    target_q_net = self.network
  end

  -- Compute max_a Q(s_2, a).
  -- We only call backward with input s, so we can use the simplified network
  q2_max = target_q_net:simple_forward(s2):float():max(2)

  target_q_net:simple_forward(s2)
  --self.next_f1 = nil
  --print(target_q_net)
  --print(target_q_net.network:get(1):get(2):get(2))
  --print(target_q_net.network:get(1):get(2):get(2).output:size())
  f1 = target_q_net:get_f1()
  self.next_f1 = target_q_net:get_f1()
  target_q_net:save_f1(self.next_f1)
  self.network:save_f1(self.next_f1)

  -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
  q2 = q2_max:clone():mul(self.discount):cmul(term)

  delta = r:clone():float()

  if self.rescale_r then
    delta:div(self.r_max)
  end
  delta:add(q2)

  -- q = Q(s,a)
  local q_all
  if not backprop then
    q_all = self.network:simple_forward(s):float()
  else
    action_tensor = self:makeActionTensor(a)
    s = s:cat(action_tensor)
    q_all = self.network:forward(s):float()
  end
  q = torch.FloatTensor(q_all:size(1))
  for i=1,q_all:size(1) do
    q[i] = q_all[i][a[i]]
  end
  delta:add(-1, q)

  if self.clip_delta then
    delta[delta:ge(self.clip_delta)] = self.clip_delta
    delta[delta:le(-self.clip_delta)] = -self.clip_delta
  end

  local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
  --print("Num actions is: ", a:size(1))
  --print()
  for i=1,math.min(self.minibatch_size,a:size(1)) do
    targets[i][a[i]] = delta[i]
  end

  if self.gpu >= 0 then targets = targets:cuda() end

  return targets, delta, q2_max, s
end

-- Assume self.dw has been computed; check the ith dimension against a computation
function nql:gradCheck(i, s, s2)
  local h = 1e-5
  hv = torch.zeros(s:size())
  hv:select(2,i):copy(torch.ones(hv:select(2,i):size()):mul(h))
  splus = s + hv
  sminus = s -hv
  if i < self.state_dim then
    s2plus = s2
    s2minus = s2
  else
    j = self.state_dim + i
    hv = torch.zeros(s2:size())
    hv:select(2,j):copy(torch.ones(hv:select(2,j):size()):mul(h))
    s2plus = s2 + hv
    s2minus = s2 -hv
  end

  local tp, dp, q2_maxp, sp = self:getQUpdate{s=splus, a=a, r=r, s2=s2plus,
  term=term, update_qmax=true, backprop=true}
  local tn, dn, q2_maxn, sn = self:getQUpdate{s=sminus, a=a, r=r, s2=s2minus,
  term=term, update_qmax=true, backprop=true}
  grad_est = (dp - dn ) / 2*h

  rel_err = torch.Tensor(self.minibatch_size)
  for m = 1, self.minibatch_size do
    rel_err[m] = torch.abs( grad_est[i] - self.dw[i]) / torch.max(torch.abs(grad_est[i]), torch.abs(self.dw[i]))
  end
  print(rel_err)
end

function nql:qLearnMinibatch(with_enaction)
  -- Perform a minibatch Q-learning update:
  -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
  assert(self.transitions:size() > self.minibatch_size)

  local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

  -- Can we replace backprop=true with backprop=with_enaction?  Only
  -- potentially helps efficiency, so we'll come back to this
  -- question later.

  -- if with_enaction then
  --    --print("Set up all the aux variables here")
  -- else
  --    self.w, self.dw = self.network.simple_network:getParameters()  -- debugging only
  --    self.dw:zero()

  --    --I think these next variables actually want to persist.
  --    --self.deltas = self.dw:clone():fill(0)
  --    --self.tmp= self.dw:clone():fill(0)
  --    --self.g  = self.dw:clone():fill(0)
  --    --self.g2 = self.dw:clone():fill(0)
  -- end


  local targets, delta, q2_max, s = self:getQUpdate{s=s, a=a, r=r, s2=s2,
  term=term, update_qmax=true, backprop=true}

  --setting backprop to false for now to keep everything in the
  --simple network.  I want to see if we can just get this network to
  --learn normally!  This should be changed back when we reintroduce
  --trying to learn the self-representation.

  -- We'll set the network parameters to correspond to the part of the network we're working with.  This is where we'll switch.
  if (self.numSteps - self.learn_start)  <= 10 then
    print("Setting weight handles to simple_network weights")
    self.w, self.dw = self.network.simple_network:getParameters()
    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)
  end
  -- zero gradients of parameters
  self.dw:zero()

  -- get new gradient
  --with_enaction=false

  -- right now with_enaction is enaction only.  this isn't inherenty
  -- unreasonable, but we probably don't want it to be the bulk of
  -- the training.
  if with_enaction then
    prediction = self.network.prednet.output:clone()

    --This shouldn't be necessary, but E-loss isn't changing
    self.w, self.dw = self.network.pred_training_network:getParameters()
    self.dw:zero()
    --delete blcok after debug

    gradOut = torch.Tensor(self.minibatch_size, 2, 7, 7):cuda() -- hardcoding 7
    gradOutP = gradOut:select(2,1)
    gradOutC = gradOut:select(2,2)


    -- Writing gradOutC = (self.network:get_f1() - prediction):mul(self.eta):cuda()
    -- doesn't actually modify gradOut; we need to do this the torch way.
    gradOutC:copy(self.network:get_f1())
    gradOutC:csub(self.network.prednet.output):mul(self.eta * 0.0001)
    -- gradOutP:mul(0):csub(gradOutC) -- mul(0) makes cuda unhappy
    gradOutP:zero():csub(gradOutC) -- mul(0) makes cuda unhappy

    action_tensor = self:makeActionTensor(a)
    s = s:cat(action_tensor):cuda()

    print("w, dw sum before: " .. self.w:sum() .. "  " .. self.dw:sum())
    print("f1 and prediction sums are:  " .. self.network:get_f1():sum() .. "  and   " .. prediction:sum())
    print("gradOutC, gradOutP, gradOut sums before: " .. gradOutC:sum() .. " " .. gradOutP:sum() .. " " .. gradOut:sum())

    self.network.pred_training_network:backward(s, gradOut)

    print("w, dw sum after: " .. self.w:sum() .. "  " .. self.dw:sum())
    print("gradOutC, gradOutP, gradOut sums after: " .. gradOutC:sum() .. " " .. gradOutP:sum() .. " " .. gradOut:sum())
    print()
    --print("dw is:  ")
    --print(self.dw)

    -- The output J (and thus dD) of the main_network will be a tensor
    -- of size minibatch_size x (num_conv_features + 1 + 1) x 7 x 7 (at
    -- least for the current config).

    -- The first num_conv_features in dimension 2 correspond to the
    -- conv_net, the penultimate feature of the output is f1, and the
    -- last is the prediction.
    -- JB:  This has changed from before.
    --dD:select(2,9):add(dC)
    --dD:select(2,10):add(dP)
    --self.main_network:updateGradInput(input, dD)
    --self.network.main_network:backward(s, targets)
  else
    --print("Non-enaction backward.")

    self.network.simple_network:backward(s, targets)
    -- if self.dw:sum() == 0 then
    -- 	 print("Got a zero dw after simple_network:backward!")
    -- end
    --print("Sum of simple weights (step " .. self.num_steps .. "), dw after update are " .. self.w:sum() .. " " .. self.dw:sum())
    --print("Sum of simple weights, dw after update are " .. self.snw:sum() .. " " .. self.sndw:sum())
    --print("Sum of differential for w, dw is " .. (self.w:sum() - snw:sum()) .. "   " .. (self.dw:sum() - sndw:sum()))
  end

  if self.numSteps % 10000 == 0 then
    evg = 0
    prediction = self.network.prednet.output:select(2,1)
    A = (self.network:get_f1() - prediction):mul(self.eta)
    l, w = A:size()[2], A:size()[3]
    for m = 1, self.minibatch_size do
      em = A[m]:reshape(1, l*w)
      evg = evg + em:pow(2):sum()
    end
    evg = evg / self.minibatch_size
    print()
    print("Average loss from Q is:  ", delta:mean())
    print("Average loss from E is:", evg)
    print("Eta is: ", self.eta)
  end

  --self.w:add(-self.lr*self.eta, self.dw)

  -- add weight cost to gradient  (regularization)
  self.dw:add(-self.wc, self.w)

  -- compute linearly annealed learning rate
  local t = math.max(0, self.numSteps - self.learn_start)
  self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
  self.lr_end
  self.lr = math.max(self.lr, self.lr_end)

  -- use gradients
  -- Formula for RMSProp from:  http://cs231n.github.io/neural-networks-3/
  --   cache = decay_rate * cache + (1 - decay_rate) * dx**2
  --   x += - learning_rate * dx / (np.sqrt(cache) + eps)

  -- Formula here:  g <- 0.95g + 0.05dw; g2 <- .95g2 + 0.05 (dw^2);
  -- So g2 <--> cache
  -- tmp <- sqrt( (g2 - g^2) + 0.001)
  self.g:mul(0.95):add(0.05, self.dw)
  self.tmp:cmul(self.dw, self.dw)
  self.g2:mul(0.95):add(0.05, self.tmp)
  self.tmp:cmul(self.g, self.g)
  self.tmp:mul(-1)
  self.tmp:add(self.g2)
  self.tmp:add(0.01)
  self.tmp:sqrt()

  -- accumulate update

  --the next line can be commented out to switch to sgd.
  self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)


  -- debugging
  print("tmp, deltas, dw sums are " .. self.tmp:sum() .. "  " .. self.deltas:sum() .. "   " .. self.dw:sum())
  print("w sum (before) is " .. self.w:sum())

  --the next line can be uncommented in to switch to sgd.
  --self.deltas:mul(0):add(-self.lr, self.dw)
  self.w:add(self.deltas)


  -- debugging
  print("w sum (after) is " .. self.w:sum())
end


function nql:sample_validation_data()
  local s, a, r, s2, term = self.transitions:sample(self.valid_size)
  self.valid_s    = s:clone()
  self.valid_a    = a:clone()
  self.valid_r    = r:clone()
  self.valid_s2   = s2:clone()
  self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
  local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
  a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term, backprop=false}

  self.v_avg = self.q_max * q2_max:mean()
  self.tderr_avg = delta:clone():abs():mean()

  -- Also compute "enaction" error.
  --prediction = self.network.prednet.output:clone()
  --E = (self.next_f1 - prediction):pow(2):sum(2):sum(3)
  --self.enaction_err_avg = E:view(E:nElement()):mean()
end

function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
  -- Preprocess state (will be set to nil if terminal)
  local state = self:preprocess(rawstate):float()
  local curState

  if self.max_reward then
    reward = math.min(reward, self.max_reward)
  end
  if self.min_reward then
    reward = math.max(reward, self.min_reward)
  end
  if self.rescale_r then
    self.r_max = math.max(self.r_max, reward)
  end

  self.transitions:add_recent_state(state, terminal)

  local currentFullState = self.transitions:get_recent()

  --Store transition s, a, r, s'
  if self.lastState and not testing then
    self.transitions:add(self.lastState, self.lastAction, reward,
      self.lastTerminal, priority)
  end

  if self.numSteps == self.learn_start+1 and not testing then
    self:sample_validation_data()
  end

  curState= self.transitions:get_recent()
  curState = curState:resize(1, unpack(self.input_dims))

  -- Select action
  local actionIndex = 1
  if not terminal then
    actionIndex = self:eGreedy(curState, testing_ep)
  end

  self.transitions:add_recent_action(actionIndex)

  --Do some Q-learning updates
  --Try alternating between optimizing for enactive part of network and simple network.
  if self.numSteps > self.learn_start then
    if (self.numSteps % self.self_interval) == 0 then
      if (not self.enactiveLearning) then
        print("Turning on self learning at step " .. self.numSteps .. " (testing is " .. tostring(testing) ..") " )
      end
      self.enactiveLearning = true

      self.w, self.dw = self.network.pred_training_network:getParameters()
      self.dw:zero()

      self.deltas = self.dw:clone():fill(0)
      self.tmp= self.dw:clone():fill(0)
      self.g  = self.dw:clone():fill(0)
      self.g2 = self.dw:clone():fill(0)

    end

    if (self.enactiveLearning == true) and (self.numSteps % self.self_interval == self.self_period) then
      self.enactiveLearning = false
      self.w, self.dw = self.network.simple_network:getParameters()
      self.dw:zero()

      self.deltas = self.dw:clone():fill(0)
      self.tmp= self.dw:clone():fill(0)
      self.g  = self.dw:clone():fill(0)
      self.g2 = self.dw:clone():fill(0)

      print("Turning off self learning at step " .. self.numSteps)
    end
  end

  if self.numSteps > self.learn_start and not testing and self.numSteps % self.update_freq == 0 then
    for i = 1, self.n_replay do
      self:qLearnMinibatch(self.enactiveLearning)
    end
  end

  if not testing then
    self.numSteps = self.numSteps + 1
  end

  self.lastState = state:clone()
  self.lastAction = actionIndex
  self.lastTerminal = terminal

  if self.target_q and self.numSteps % self.target_q == 1 then
    self.target_network = self.network:clone()
  end

  if not terminal then
    return actionIndex
  else
    return 0
  end
end


function nql:eGreedy(state, testing_ep)
  self.ep = testing_ep or (self.ep_end +
    math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
    math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
  -- Epsilon greedy
  if torch.uniform() < self.ep then
    return torch.random(1, self.n_actions)
  else
    return self:greedy(state)
  end
end


function nql:greedy(state)
  -- Turn single state into minibatch.  Needed for convolutional nets.

  -- JB: I don't think the architecture is configured like this
  -- anymore; we're expecting a minibatch of single states which the
  -- network will narrow and resize as needed.

  -- if state:dim() == 2 then
  --    assert(false, 'Input must be at least 3D')
  --    state = state:resize(1, state:size(1), state:size(2))
  -- end

  -- JB: In fact, the state seems to come in as a 4-tensor and we
  -- want a 2-tensor.  Specifically, it comes in with dimensions
  -- (#minibatches, hist_len, screen_len, screen_width); we will
  -- collapse the last three dimensions.

  if state:dim() == 4 then
    state = state:resize(state:size(1), state:size(2)*state:size(3)*state:size(4))
  end

  if state:dim() == 3 then
    assert(false, 'Input must be 2D')
  end

  if self.gpu >= 0 then
    state = state:cuda()
  end

  local q = self.network:simple_forward(state):float():squeeze()
  local maxq = q[1]
  local besta = {1}

  -- Evaluate all other actions (with random tie-breaking)
  for a = 2, self.n_actions do
    if q[a] > maxq then
      besta = { a }
      maxq = q[a]
    elseif q[a] == maxq then
      besta[#besta+1] = a
    end
  end
  self.bestq = maxq

  local r = torch.random(1, #besta)

  self.lastAction = besta[r]

  return besta[r]
end


function nql:createNetwork()
  local n_hid = 128
  local mlp = nn.Sequential()
  mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
  mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
  mlp:add(nn.Rectifier())
  mlp:add(nn.Linear(n_hid, n_hid))
  mlp:add(nn.Rectifier())
  mlp:add(nn.Linear(n_hid, self.n_actions))

  return mlp
end


function nql:_loadNet()
  local net = self.network
  if self.gpu then
    net:cuda()
  else
    net:float()
  end
  return net
end


function nql:init(arg)
  self.actions = arg.actions
  self.n_actions = #self.actions
  self.network = self:_loadNet()
  -- Generate targets.
  self.transitions:empty()
end


function nql:report()
  print(get_weight_norms(self.network))
  print(get_grad_norms(self.network))
end

