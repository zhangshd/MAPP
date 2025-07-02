'''
Author: zhangshd
Date: 2024-09-09 21:36:29
LastEditors: zhangshd
LastEditTime: 2024-09-10 14:28:33
'''

import os,sys
from pathlib import Path
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GCMC.utils import run_simulation



async def main(test_dir):
    
    # Get the number of CPUs
    cpu_count = max(os.cpu_count() - 2, 1)
    # Create a Semaphore, limiting the concurrency to the number of CPUs
    semaphore = asyncio.Semaphore(cpu_count)

    async def run_task(task_dir):
        # Use Semaphore to limit concurrency
        async with semaphore:
            await run_simulation(task_dir)

    
    with open(test_dir/"failed_records.txt") as f:
        failed_dirs = f.read().split("\n")

    processes = []
    for task_dir in failed_dirs:
        processes.append(
                run_task(task_dir)
                )
    await asyncio.gather(*processes)


if __name__ == "__main__":
    main_dir = Path("/home/zhangsd/repos/MOF-MTHNN")

    test_dir = main_dir/"GCMC/gem_190_313K_cutoff_truncated"

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(test_dir))