# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:06:19 2020

@author: Attitude
"""

# Audio To Text (Speech Recognition)

# Import Library
import os
import speech_recognition as sr
sr.__version__

# Speech Recognition starts here
r = sr.Recognizer()

current_dir = os.path.dirname(os.path.realpath(__file__))
test_dir = os.path.join(current_dir, 'test_sound_mono')
fp = os.path.join(test_dir, 'datang_2.wav')

aud = sr.AudioFile(fp)
with aud as source:
    audio = r.record(source)

r.recognize_google(audio)

# This is for checking audio variable type
type(audio)