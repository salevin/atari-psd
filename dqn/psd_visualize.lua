--require 'image'
--require 'gnuplot'
require 'unsup'
require 'graph'
require 'cunn'
require 'image'

dofile 'psd_data.lua'

if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Visualizing a PSD Model')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-reload','', 'mlp bin file')
cmd:text()

local params = cmd:parse(arg)

if not paths.filep(params.reload) then
  error('Hey, Listen!\nYou must specify a .bin mlp!')
end

mlp = torch.DiskFile(params.reload,'r'):binary():readObject()

-- create the dataset
data = getdata(500)--imutils.mapdata(params.data)
data:conv()

print("Encoder:")
print(mlp.encoder)

print("Decoder:")
print(mlp.decoder.D.weight)

print("Prediction weight:")
print(mlp.beta)

function sleep(n)
  os.execute("sleep " .. tonumber(n))
  print("not sleeping")
end

for i=1,100 do
  ex = data[i]
  input = ex[1]
  target = ex[2]


  res = mlp.encoder:forward(input, target)

  c = image.toDisplayTensor({input=res,nrow=math.ceil(math.sqrt(res:size(1))),symmetric=true,padding=1})
  w1 = image.display({image=c, win=w1})


  c = image.toDisplayTensor({input=input,nrow=math.ceil(math.sqrt(input:size(1))),symmetric=true,padding=1})
  w2 = image.display({image=c, win=w2})

  sleep(2)
end

input1 = nn.Identity()()
input2 = mlp.encoder(input1)

g = nn.gModule({input1}, {input2})
g:forward(input, target)

graph.dot(g.fg, 'Forward Graph')
--graph.dot(mlp.encoder.fg, 'MLP')
