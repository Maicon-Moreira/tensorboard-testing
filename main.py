import torch as t
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# delete logs folder
if os.path.exists("./logs"):
    shutil.rmtree("./logs")


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x


simple_model = SimpleModel()


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

    # graph
    writer.add_graph(simple_model, (t.rand(1, 10),))

    # embedding
    embedding = t.rand(10, 10)
    meta = [str(i) for i in range(10)]
    writer.add_embedding(embedding, metadata=meta)

    # pr_curve
    labels = np.random.randint(2, size=100)  # binary label
    predictions = np.random.rand(100)
    writer.add_pr_curve("pr_curve", labels, predictions, 0)

    # custom scalars
    layout = {
        "Taiwan": {"twse": ["Multiline", ["twse/0050", "twse/2330"]]},
        "USA": {
            "dow": ["Margin", ["dow/aaa", "dow/bbb", "dow/ccc"]],
            "nasdaq": ["Margin", ["nasdaq/aaa", "nasdaq/bbb", "nasdaq/ccc"]],
        },
    }
    writer.add_custom_scalars(layout)

    # mesh
    vertices_tensor = t.as_tensor(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
        ],
        dtype=t.float,
    ).unsqueeze(0)
    colors_tensor = t.as_tensor(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 255],
        ],
        dtype=t.int,
    ).unsqueeze(0)
    faces_tensor = t.as_tensor(
        [
            [0, 2, 3],
            [0, 3, 1],
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=t.int,
    ).unsqueeze(0)
    writer.add_mesh(
        "my_mesh", vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor
    )

    # writer.add_scalar("maicon", 0.5, 0)

    writer.close()
