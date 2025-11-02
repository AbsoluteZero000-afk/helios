"""
Helios Main Entry

Initializes logging and can perform a quick health check.
"""

import asyncio
from utils.logger import setup_logging
from utils.sentinel import system_sentinel


def main():
    setup_logging()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(system_sentinel.start_monitoring())
    loop.run_until_complete(system_sentinel.stop_monitoring())


if __name__ == "__main__":
    main()
