"""
轻量壳子脚本：
- 所有真实逻辑都在 src/pipeline_eval_audiocaps.py 里面；
- 这里不再改写任何搜索函数，只是转发命令行参数。
"""

from pipeline_eval_audiocaps import main

if __name__ == "__main__":
    # 直接调用原始 main()，它会用 argparse 读取命令行参数，
    # 搜索 adv/noise 的逻辑也全部用原来已经跑通的实现。
    main()
