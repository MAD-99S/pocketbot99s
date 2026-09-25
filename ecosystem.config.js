module.exports = {
  apps: [
    {
      name: "pocketbot",
      script: ".venv/bin/python",
      args: "-m apps.manual_trading.main",
      cwd: "/home/ubuntu/pocketbot",
      env: {
        PYTHONPATH: "/home/ubuntu/pocketbot",
      },
      autorestart: true,
      max_memory_restart: 1073741824,
      instances: 1,
      exec_mode: "fork",
      max_restarts: 10,
      min_uptime: "5s",
      kill_timeout: 15000,
      merge_logs: true,
      error_file: "./logs/pocketbot-err.log",
      out_file: "./logs/pocketbot-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    },
  ],
};
