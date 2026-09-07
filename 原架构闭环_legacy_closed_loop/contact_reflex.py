"""Born contact withdrawal from actual tactile direction and forward depth.

This repairs the old any-contact -> reverse deadlock. It is an explicit innate
body controller, not learned navigation or a route supplied to the brain.
No coordinates, goal labels, timers, extra memory or world internals are read.
"""
import numpy as np


def contact_reflex(packet):
    observation=packet['observation']
    touch=np.asarray(observation['touch'],float)  # front, back, left, right
    distances=np.asarray(observation['ray_distances'])
    if np.max(touch)>.02:
        side=int(np.argmax(touch))
        if side in (0,1):
            turn_left=np.mean(distances[19:])>np.mean(distances[:12])
            if side==1:
                return np.array([.1,0.,.5,0.]) if turn_left else np.array([.5,0.,.1,0.])
            # A backward trajectory bends left with clockwise body rotation.
            return np.array([0.,.1,0.,.5]) if turn_left else np.array([0.,.5,0.,.1])
        return np.array([.5,0.,.1,0.]) if side==2 else np.array([.1,0.,.5,0.])
    if float(observation['pain'])>.02:
        return np.zeros(4)
    if np.min(distances[12:19])<.22:
        return np.array([0.,.5,0.,.5])
    return None
