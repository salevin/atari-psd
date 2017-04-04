--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'linnet'

return function(args)
    args.n_units        = {16*7*7}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier

    return create_network(args)
end

