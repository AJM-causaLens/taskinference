import numpy as np
# Set random seed for reproducibility
np.random.seed(42)

reacher_test_tasks = [(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)) for _ in range(10)]
reacher_train_tasks = [(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)) for _ in range(100)]


def generate_valid_point_mass_tasks(num_tasks=10):
    point_mass_test_tasks = []

    for _ in range(num_tasks):
        while True:
            task = (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
            if np.linalg.norm(task) >= 0.1:  # Ensure distance is at least 0.1 from (0,0)
                rounded_task = tuple(np.round(task, 3))
                point_mass_test_tasks.append(rounded_task)
                break  # Task is valid, exit loop

    return point_mass_test_tasks

def generate_valid_point_mass_tasks_structured(num_tasks=10, radius=0.2, seed=None):
    """
    """
    if seed is not None:
        np.random.seed(seed)

    # Randomly select angles between 0 and 2π
    angles = np.random.uniform(0, 2 * np.pi, num_tasks)
    tasks = [(round(radius * np.cos(angle), 3), round(radius * np.sin(angle), 3))
             for angle in angles]
    return tasks

point_mass_train_tasks = generate_valid_point_mass_tasks(100)
point_mass_test_tasks = generate_valid_point_mass_tasks(20)

point_mass_train_easy = generate_valid_point_mass_tasks_structured(100, 0.1, 42)
point_mass_test_easy = generate_valid_point_mass_tasks_structured(20, 0.1, 84)

point_mass_train_mid = generate_valid_point_mass_tasks_structured(100, 0.15, 42)
point_mass_test_mid = generate_valid_point_mass_tasks_structured(20, 0.15, 84)

point_mass_train_hard = generate_valid_point_mass_tasks_structured(100, 0.2, 42)
point_mass_test_hard = generate_valid_point_mass_tasks_structured(20, 0.2, 84)


cheetah_run_train = [[np.random.uniform(2, 4)] for _ in range(50)]
cheetah_run_test = [np.random.uniform(2, 4) for _ in range(5)]

cheetah_train_easy, cheetah_test_easy = [np.random.uniform(2, 4) for _ in range(50)], [np.random.uniform(2, 4) for _ in range(5)]
cheetah_train_mid, cheetah_test_mid = [np.random.uniform(4, 7) for _ in range(50)], [np.random.uniform(4, 7) for _ in range(5)]
cheetah_train_hard, cheetah_test_hard = [np.random.uniform(7, 10) for _ in range(50)], [np.random.uniform(7, 10) for _ in range(5)]
cheetah_train_all, cheetah_test_all = [(np.random.uniform(0.5, 3)) for _ in range(50)], [(np.random.uniform(0.5, 3)) for _ in range(5)]
task_dict = {"reacher_easy": (reacher_train_tasks, reacher_test_tasks),
             "point_mass_easy": (point_mass_train_tasks, point_mass_test_tasks),
             "cheetah_run": (cheetah_run_train, cheetah_run_test),
             "HalfCheetah-v5": (cheetah_train_easy, cheetah_test_easy),
             "cheetah_easy": (cheetah_train_mid, cheetah_test_mid),
             "cheetah_mid": (cheetah_run_train, cheetah_run_test),
             "cheetah_hard": (cheetah_train_hard, cheetah_test_hard),
             "cheetah_all": (cheetah_train_all, cheetah_test_all),
             "point_mass_close": (point_mass_train_easy, point_mass_test_easy),
             "point_mass_mid": (point_mass_train_mid, point_mass_test_mid),
             "point_mass_far": (point_mass_train_hard, point_mass_test_hard),}
