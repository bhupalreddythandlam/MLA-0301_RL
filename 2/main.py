"""Command-line runner for all assignment tasks."""
import sys
from task1_mdp import print_task1
from task2_dynamic_programming import run_task2
from task3_monte_carlo_td import run_task3
from task4_td_control import run_task4
from task5_warehouse_project import run_task5

TASKS = {
    "task1": print_task1,
    "task2": run_task2,
    "task3": run_task3,
    "task4": run_task4,
    "task5": run_task5,
}

def main():
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if choice == "all":
        for function in TASKS.values():
            function()
            print("\n" + "=" * 65 + "\n")
    elif choice in TASKS:
        TASKS[choice]()
    else:
        print("Usage: python main.py [all|task1|task2|task3|task4|task5]")

if __name__ == "__main__":
    main()
