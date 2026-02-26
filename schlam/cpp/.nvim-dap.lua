local dap = require("dap")

-- This defines the C++ debug configuration for THIS project
dap.configurations.cpp = {
  {
    name = "Debug My Project",
    type = "codelldb", -- Ensure this matches your Mason adapter name
    request = "launch",
    -- Point directly to your compiled binary
    program = function()
      return vim.fn.getcwd() .. "/build/Debug/app"
    end,
    -- Hardcode your arguments here so you don't have to type them
    args = {  },
    cwd = "${workspaceFolder}",
    initCommands = {
      "settings set target.disable-aslr false"
    },
    stopOnEntry = false,
    -- You can also set Environment Variables here
    env = function()
      return {
        LD_LIBRARY_PATH = "/usr/local/lib",
        DEBUG_LOG = "true",
      }
    end,
  },
}
