"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import namedtuple

import gradio as gr
import numpy as np
import os


ModelInfo = namedtuple('ModelInfo',
                       ['name', 'face_args', 'pose_args', 'face_model', 'pose_model', 'render_name', 'pose_resume_trans'])


def file_exists(filepath: str) -> bool:
    ok = os.path.exists(filepath)
    print(f"{filepath} - {ok}")
    if not ok:
        gr.Error(f"{filepath} not exists")

    return ok


class GradioModel:
    def __init__(self, info: ModelInfo) -> None:
        self.model_info = info

        self._setup_model(
            info.face_args, info.face_model,
        )
        self._setup_model(
            info.pose_args, info.pose_model,
        )
        # load standardization stuff
        file_exists(f"dataset/{info.render_name}/data_stats.pth")
        # set up renderer
        config_base = f"./checkpoints/ca_body/data/{info.render_name}"
        file_exists(config_base)

    def _setup_model(
            self,
            args_path: str,
            model_path: str,
    ):
        file_exists(args_path)
        file_exists(model_path)
        file_exists(self.model_info.pose_resume_trans)


def audio_to_avatar(audio: np.ndarray, num_repetitions: int, top_p: float, actor: str, render: str):
    global gradio_model

    model_info_list = [
        ModelInfo(
            name="Actor 1",
            face_args='./checkpoints/diffusion/c1_face/args.json',
            pose_args='./checkpoints/diffusion/c1_pose/args.json',
            face_model='checkpoints/diffusion/c1_face/model000155000.pt',
            pose_model='checkpoints/diffusion/c1_pose/model000340000.pt',
            render_name=render,
            pose_resume_trans="checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt",
        ),

        ModelInfo(
            name="Actor 2",
            face_args='./checkpoints/diffusion/c2_face/args.json',
            pose_args='./checkpoints/diffusion/c2_pose/args.json',
            face_model='checkpoints/diffusion/c2_face/model000190000.pt',
            pose_model='checkpoints/diffusion/c2_pose/model000190000.pt',
            render_name=render,
            pose_resume_trans="checkpoints/guide/c2_pose/checkpoints/iter-0135000.pt",
        ),

        ModelInfo(
            name="Actor 3",
            face_args='./checkpoints/diffusion/c3_face/args.json',
            pose_args='./checkpoints/diffusion/c3_pose/args.json',
            face_model='checkpoints/diffusion/c3_face/model000180000.pt',
            pose_model='checkpoints/diffusion/c3_pose/model000455000.pt',
            render_name=render,
            pose_resume_trans="checkpoints/guide/c3_pose/checkpoints/iter-0210000.pt",
        ),

        ModelInfo(
            name="Actor 4",
            face_args='./checkpoints/diffusion/c4_face/args.json',
            pose_args='./checkpoints/diffusion/c4_pose/args.json',
            face_model='checkpoints/diffusion/c4_face/model000185000.pt',
            pose_model='checkpoints/diffusion/c4_pose/model000350000.pt',
            render_name=render,
            pose_resume_trans="checkpoints/guide/c4_pose/checkpoints/iter-0135000.pt",
        )
    ]

    try:
        actor_index = ["Actor 1", "Actor 2", "Actor 3", "Actor 4"].index(actor)
    except Exception as e:
        actor_index = 0

    feed = model_info_list[actor_index]

    gradio_model = GradioModel(feed)

    return []


demo = gr.Interface(
    audio_to_avatar,  # function
    [
        gr.Audio(sources=["upload", "microphone"]),
        gr.Number(
            value=3,
            label="Number of Samples (default = 3)",
            precision=0,
            minimum=1,
            maximum=10,
        ),
        gr.Number(
            value=0.97,
            label="Sample Diversity (default = 0.97)",
            precision=None,
            minimum=0.01,
            step=0.01,
            maximum=1.00,
        ),
        gr.Dropdown(
            ["Actor 1", "Actor 2", "Actor 3", "Actor 4"], label="Actor", info="Select Actor to play with!",
        ),
        gr.Dropdown(
            ["GQS883", "PXB184", "RLW104", "TXB805"], label="Render", info="Select Render",
        )
    ],  # input type
    [],
    # [gr.Video(format="mp4", visible=True)]
    # + [gr.Video(format="mp4", visible=False) for _ in range(9)],  # output type
    title='"From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations" Demo',
    description="You can generate a photorealistic avatar from your voice! <br/>\
        1) Start by recording your audio.  <br/>\
        2) Specify the number of samples to generate.  <br/>\
        3) Specify how diverse you want the samples to be. This tunes the cumulative probability in nucleus sampling: 0.01 = low diversity, 1.0 = high diversity.  <br/>\
        4) Then, sit back and wait for the rendering to happen! This may take a while (e.g. 30 minutes) <br/>\
        5) After, you can view the videos and download the ones you like.  <br/>",
    article="Relevant links: [Project Page](https://people.eecs.berkeley.edu/~evonne_ng/projects/audio2photoreal)",
    # TODO: code and arxiv
)

if __name__ == "__main__":
    gradio_model: GradioModel = None

    # fixseed(10)
    # demo.launch(share=True)
    demo.launch()
