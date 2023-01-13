import p5
import numpy as np

from abc import abstractmethod, ABCMeta
from enum import Enum

from ..utils.keyhandler import KeyListener


class VisualizerState(Enum):
    INITIAL = 'INITIAL'
    LOOP = 'LOOP'


class Visualizer(KeyListener):
    WIDTH = 640
    HEIGHT = 640

    @property
    def DEFAULT_FONT(self):
        return p5.create_font('Arial.ttf')

    def __init__(self):
        self.is_looping = False
        self.state = (VisualizerState.INITIAL, 0)

        self.states = {
            VisualizerState.INITIAL: [
                self._background
            ],
            VisualizerState.LOOP: [
                self._loop
            ]
        }

    def _background(self):
        p5.size(self.WIDTH, self.HEIGHT)
        p5.background(0)

        p5.stroke(255)
        p5.fill(255)
        p5.stroke_weight(1)
        p5.text_font(self.DEFAULT_FONT, size=15)

        p5.text('> for next frame, space for autoplay', self.WIDTH // 3.2, 0)

    @abstractmethod
    def _update(self):
        pass

    def _loop(self):
        if self.is_looping:
            self._update()

        self.states[VisualizerState.INITIAL][-1]()

    def key_pressed(self, key: str):
        if key == 'RIGHT':
            if self.is_looping:
                return

            state, stage = self.state
            stage += 1

            if state == VisualizerState.INITIAL and \
               stage == len(self.states[VisualizerState.INITIAL]):
                self.state = (VisualizerState.LOOP, 0)
                self._update()
                return
            elif state == VisualizerState.LOOP:
                self._update()

            self.state = (state, stage % len(self.states[state]))
        elif key == ' ':
            self.is_looping = not self.is_looping


    def draw(self):
        state, stage = self.state

        self.states[state][stage]()
