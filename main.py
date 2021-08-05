import torch as t
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


# delete logs folder
if os.path.exists("./logs"):
    shutil.rmtree("./logs")


for run in range(1, 4):
    # configuration
    run_name = "run_{}".format(run)
    log_dir = "./logs/" + run_name
    writer = SummaryWriter(log_dir=log_dir)
    print(log_dir)

    # scalar, scalars, histogram
    for i in range(10):
        sleep(0.1 / run)
        writer.add_scalar("test_scalar", i * run, i)
        writer.add_scalar("equal_scalar", i, i)
        writer.add_scalars(
            "scalar_group",
            {"square_x": i ** 2 * run, "cubic_x": i ** 3 * run},
            i,
        )
        array = np.random.random(100)
        writer.add_histogram("test_histogram", array + i, i)

    # image
    image = t.rand(3, 100, 100)
    writer.add_image("test_image", image)

    # figure
    figure = plt.figure()
    plt.plot([i for i in range(100)], [i * run for i in range(100)])
    writer.add_figure("test_figure", figure)

    # video
    video = t.rand(1, 50, 3, 10, 10)
    writer.add_video("test_video", video, fps=10)

    # audio
    audio = t.rand(1, 100000, 1)
    writer.add_audio("test_audio", audio)

    # text
    text = f"Hello world from run {run}"
    writer.add_text("test_text", text)

    writer.close()