require "xlua"
require "initenv"


function parse_options()
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Train Agent in Environment:')
   cmd:text()
   cmd:text('Options:')
   
   cmd:option('-framework', 'alewrap', 'name of training framework')
   cmd:option('-env', 'breakout', 'name of environment to use')
   cmd:option('-game_path', '../roms/', 'path to environment file (ROM)')
   cmd:option('-env_params', 'useRGB=true', 'string of environment parameters')
   cmd:option('-pool_frms', 'type="max"',
	      'string of frame pooling parameters (e.g.: size=2,type="max")')
   cmd:option('-actrep', 4, 'how many times to repeat action')
   cmd:option('-random_starts', 30, 'play action 0 between 1 and random_starts ' ..
		 'number of times at the start of each training episode')
   
   cmd:option('-name', 'ecn_breakout_FULL_Y', 'filename used for saving network and training history')
   cmd:option('-network', '"convnet_atari3"', 'reload pretrained network')
   cmd:option('-agent', 'NeuralQLearner', 'name of agent file to use')
   cmd:option('-agent_params', 'lr=0.00025,ep=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=3,learn_start=10000,replay_memory=1000000,update_freq=4,n_replay=1,network="convnet_atari3",preproc="net_downsample_2x_full_y",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1', 'string of agent parameters')
   cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
   cmd:option('-saveNetworkParams', true,
	      'saves the agent network in a separate file')
   cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
   cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
   cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
   cmd:option('-save_versions', 0, '')

   cmd:option('-steps', 10^5, 'number of training steps to perform')
   cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

   cmd:option('-verbose', 2,
	      'the higher the level, the more information is printed to screen')
   cmd:option('-threads', 1, 'number of BLAS threads')
   cmd:option('-gpu', 0, 'gpu flag')
   cmd:option('-reload', "", 'saved model to start with')
   cmd:option('-full_display', false, 'display ongoing game while collecting data and training?')
   cmd:text()

   local opt = cmd:parse(arg)
   return opt
end

function display_ongoing_weights(agent, old_screen)
  new_net = agent.network

  -- new_net = net:clone()
  -- new_net:remove()
  -- new_net:remove()
  -- new_net:remove()
  -- new_net:remove()
  -- new_net:remove()
  --new_net:float()
  local pstate  = agent:preprocess(old_screen)
  frp = torch.Tensor(2*84*84)
  frp = torch.repeatTensor(pstate, 2)
  new_net:forward(frp:cuda())
  w_state = new_net:get(6).output
  w_win = image.display({image=pstate:reshape(84,84), win=w_win})
  disp = w_state:sub(1,64)
  p_win = image.display({image=disp, win=p_win})
end


function add_one_to_trans()
  s = torch.ByteTensor(agent.transitions.stateDim):fill(0)
  p = agent:preprocess(screen)
  s = p:clone():float():mul(255):int():float():div(255)
  print("Action index is ", action_index, "Screen sum (preprocessed) is ", p:sum() , "converted is",s:sum() )
  action_index = agent:perceive(reward, screen, terminal)
  screen, reward, terminal = game_env:step(game_actions[action_index], true)
end

function add_to_trans(n)
  action_index = nil
  for i=1,n do
    add_one_to_trans()
  end
end

function init()
  opt = parse_options()
  --opt.gpu=-1
  game_env, game_actions, agent, opt = setup(opt)
  screen, reward, terminal = game_env:getState()

  --learn_start = agent.learn_start
  start_time = sys.clock()
  reward_counts = {}
  episode_counts = {}
  time_history = {}
  v_history = {}
  qmax_history = {}
  td_history = {}
  reward_history = {}
  step = 0
  time_history[1] = 0


  p_win, w_win  = nil
end

function load_agent(filename_base)
  filename = filename_base .. ".t7"
  params_file = filename_base .. ".params.t7"
  savedData = torch.load(filename)
  agent = savedData.agent

  loadWeights = torch.load(params_file, 'ascii')
  agent.w = loadWeights["network"]:cuda()

  --Not sure this is what we want long term.  Right now we're
  --breaking because numSteps is saved and it's expecting a
  --non-existent transition table.
  v, dw = agent.network:getParameters()
  agent.dw = dw
  agent:reset()
  agent.dw:zero()
  agent.numSteps = 0
  agent.lastState = nil
  agent.lastAction = nil
  agent.v_avg = 0 -- V running average.
  agent.tderr_avg = 0 -- TD error running average.

  agent.q_max = 1
  agent.r_max = 1

  agent.w, agent.dw = agent.network:getParameters()
  agent.predw, agent.pred_dw = agent.network.prednet:getParameters()
  agent.convw, agent.conv_dw = agent.network.convnet:getParameters()

  agent.dw:zero()

  agent.deltas = agent.dw:clone():fill(0)

  agent.tmp= agent.dw:clone():fill(0)
  agent.g  = agent.dw:clone():fill(0)
  agent.g2 = agent.dw:clone():fill(0)

  if agent.target_q then
    agent.target_network = agent.network:clone()
  end

  agent.C = torch.Tensor(agent.minibatch_size, unpack(agent.network.convout_dims)):zero()
  if agent.gpu and agent.gpu >= 0 then
    agent.C = agent.C:cuda()
  else
    agent.C = agent.C:float()
  end


  local transition_args = {
    stateDim = agent.state_dim, numActions = agent.n_actions,
    histLen = agent.hist_len, gpu = agent.gpu,
    maxSize = agent.replay_memory, histType = agent.histType,
    histSpacing = agent.histSpacing, nonTermProb = agent.nonTermProb,
    bufferSize = agent.bufferSize
  }
  agent.transitions = dqn.TransitionTable(transition_args)

  agent.lastState = nil
  agent.lastAction = nil
  agent.v_avg = 0 -- V running average.
  agent.tderr_avg = 0 -- TD error running average.

  print(agent.numSteps)
  print(agent.w:size())
  print(agent.w:sum())
end


function test_forward_passes()
  main_network = agent.network.main_network
  conv_net = agent.network.convnet
  pred_net = agent.network.prednet
  dec_net = agent.network.decnet

  z = torch.Tensor(agent.network.minibatch_size, agent.network.input_size):cuda()
  m = main_network:get(1)
  m1 = m:get(1)
  znarrowed = m1:forward(z)
  m2 = m:get(2)
  --zn2 = m2:forward(znarrowed)

  m22 = m2:get(2)
  zi = znarrowed
  for i=1,10 do
    m22i = m22:get(i)
    zi = m22i:forward(zi)
    print("At i = ", i)
    print(m22i)
    print("Output size is ", zi:size())
  end



  print("Testing main network forward.")
  print(main_network:forward(z1):size())
  print("Testing simple network forward.")
  print(agent.network:simple_forward(z1):size())
  print("Testing full network forward.")
  z2 = agent.network:forward(z1)
  print(z2:size())
end

function one_step_self_backprop()
  s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
  w, dw = agent.network.pred_training_network:getParameters()
  dw:zero()
  agent.network.pred_training_network:zeroGradParameters()

  --agent.network:simple_forward(s)


  action_tensor = agent:makeActionTensor(a)
  --s = s:cat(action_tensor):cuda()
  s = s:cat(action_tensor)
  agent.network:forward(s)

  --This *shouldn't* make a difference
  gradOut = torch.Tensor(agent.minibatch_size, 2, 7, 7):cuda() -- hardcoding 7
  gradOutP = gradOut:select(2,1)
  gradOutC = gradOut:select(2,2)


  -- Writing gradOutC = (self.network:get_f1() - prediction):mul(self.eta):cuda()
  -- doesn't actually modify gradOut; we need to do this the torch way.

  f1 = agent.network:get_f1()

  --debugging, let f1 be all ones:
  --f1:copy(torch.ones(f1:size()))
  prediction = agent.network.prednet.output


  --gradOutC:copy(agent.network:get_f1())  --restoore after debugging
  gradOutC:copy(f1)

  --gradOutC:copy(torch.ones(gradOutC:size()))   --for debugging

  gradOutC:csub(agent.network.prednet.output):mul(agent.eta)
  --gradOutP = gradOutC:clone():mul(-1):cuda()
  --gradOutP:mul(0):csub(gradOutC)
  gradOutP:zero():csub(gradOutC)


  --gradOutC:zero() -- temporary for debugging
  --gradOutP:zero() -- temporary for debugging

  dw:zero()
  print("w, dw sum before: " .. w:sum() .. "  " .. dw:sum())
  print("f1 and prediction sums are:  " .. f1:sum() .. "  and   " .. prediction:sum())
  print("gradOutC, gradOutP, gradOut sums before: " .. gradOutC:sum() .. " " .. gradOutP:sum() .. " " .. gradOut:sum())
  agent.network.pred_training_network:backward(s, gradOut)
  w:add(-0.01, dw)
  print("w, dw sum after: " .. w:sum() .. "  " .. dw:sum())
  print("gradOutC, gradOutP, gradOut sums after: " .. gradOutC:sum() .. " " .. gradOutP:sum() .. " " .. gradOut:sum())
  print()

  s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
end

function bug_report()
  s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
  w, dw = agent.network.pred_training_network:getParameters()
  dw:zero()
  agent.network.pred_training_network:zeroGradParameters()

  agent.network:simple_forward(s)


  gradOut = torch.Tensor(agent.minibatch_size, 2, 7, 7) -- hardcoding 7
  gradOutP = gradOut:select(2,1)
  gradOutC = gradOut:select(2,2)
  gradOutC:copy(torch.ones(gradOutC:size()))   --for debugging

  gradOutC:csub(agent.network.prednet.output):mul(agent.eta)
  gradOutP:mul(0):csub(gradOutC)
end
function some_steps_self_backprop(n, l)
  ns = n or 1
  ls = l or 6000
  init()
  agent.learn_start = ls
  collect_data(ls)
  s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
  w, dw = agent.network.pred_training_network:getParameters()
  agent.network.pred_training_network:zeroGradParameters()
  dw:zero()
  gradOut = torch.Tensor(agent.minibatch_size, 2, 7, 7):cuda() -- hardcoding 7
  gradOutP = gradOut:select(2,1)
  gradOutC = gradOut:select(2,2)


  for i=1,ns do
    one_step_self_backprop()
    assert(w:sum() == w:sum())
  end


end


function collect_data(n)
  num_steps = n   step=0
  print("Collecting " .. num_steps .. " images.")
  collect = torch.CudaTensor(n,210,160)

  -- For debugging
  w, dw = agent.network:getParameters()
  for i=1,num_steps do
    xlua.progress(i, num_steps)
    step = step + 1
    old_screen = screen
    action_index = agent:perceive(reward, screen, terminal)

    -- game over? get next game!
    if not terminal then
      screen, reward, terminal = game_env:step(game_actions[action_index], true)
    else
      if opt.random_starts > 0 then
        screen, reward, terminal = game_env:nextRandomGame()
      else
        screen, reward, terminal = game_env:newGame()
      end
    end

    collect[i] = screen[1][1]

    -- display screen
    if full_display then
      win = image.display({image=screen, win=win})
    end

    -- display some weights (JB)
    if full_display then
      --weight_win = display_ongoing_weights(agent, old_screen)
    end
  end
  return collect
end


function debug_to_get_preds()
  full_display = true
  arg=""
  init()
  load_agent("DQN3_0_1_breakout_FULL_Y50000000")
  collect_data(60000)
  s, a, r, s2, term = agent.transitions:sample(agent.minibatch_size)
  targets, delta, q2_max, s = agent:getQUpdate{s=s, a=a, r=r, s2=s2, term=term, update_qmax=true, backprop=true}
  agent.dw:zero()
  agent.network:pred_training_updateGradients(s, targets)
end
