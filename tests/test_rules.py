"""Правила: склейка циклов без паузы, наклон головы, неправдоподобная рука, рамка события."""
import dataclasses

import numpy as np
from test_logic import make_signal, puffs

from smokedet.config import CycleConfig, EpisodeConfig, GeometryConfig
from smokedet.cycles import detect_cycles
from smokedet.episodes import build_events, event_box, qualifying_cycles
from smokedet.geometry import L_ELB, L_SHO, L_WRI, R_ELB, R_SHO, R_WRI, arm_plausible


def test_back_to_back_cycles_count_as_one_movement():
    sig = make_signal(puffs([3, 10]))
    cycles = detect_cycles(sig, CycleConfig())
    assert len(cycles) == 2 and all(c.valid for c in cycles)
    cycles[1].start = cycles[0].end  # второй подход начался в момент конца отвода — одно движение
    assert qualifying_cycles(cycles, EpisodeConfig()) == []
    assert cycles[1].joined_prev and not cycles[0].joined_prev
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig()) == []
    off = EpisodeConfig(min_cycle_gap=-1.0)
    assert len(qualifying_cycles(cycles, off)) == 2 and not cycles[1].joined_prev
    events = build_events({1: cycles}, {1: sig}, off)
    assert len(events) == 1 and events[0].to_dict()["n_movements"] == 2


def test_cycles_with_pause_between_still_count_twice():
    sig = make_signal(puffs([3, 10]))
    cycles = detect_cycles(sig, CycleConfig())
    assert len(qualifying_cycles(cycles, EpisodeConfig())) == 2


def _tilted_signal(rise):
    sig = make_signal(puffs([3]))
    pitch = np.full(len(sig), 0.3)
    pitch[(sig.t >= 3.5) & (sig.t < 5.0)] += rise  # в паузе нос поднялся над ушами
    sig.head_pitch = pitch
    return sig


def test_head_tilt_feature_and_optional_reject():
    c = detect_cycles(_tilted_signal(0.4), CycleConfig())[0]
    assert abs(c.head_tilt - 0.4) < 1e-6 and c.valid  # по умолчанию фильтр выключен
    assert c.to_dict()["head_tilt"] == 0.4
    c = detect_cycles(_tilted_signal(0.4), CycleConfig(max_head_tilt=0.3))[0]
    assert not c.valid and "head_tilt" in c.reject_reason
    c = detect_cycles(_tilted_signal(0.1), CycleConfig(max_head_tilt=0.3))[0]
    assert c.valid
    sig = make_signal(puffs([3]))  # нет точек головы — признак nan, цикл не трогается
    c = detect_cycles(sig, CycleConfig(max_head_tilt=0.3))[0]
    assert np.isnan(c.head_tilt) and c.valid


def _kpts(l_elb=(100, 130), l_wri=(100, 160)):
    k = np.zeros((17, 3), np.float32)
    for j, xy in ((L_SHO, (100, 100)), (R_SHO, (140, 100)), (L_ELB, l_elb), (R_ELB, (140, 130)),
                  (L_WRI, l_wri), (R_WRI, (140, 160))):  # fmt: skip
        k[j] = (*xy, 0.9)
    return k


def test_arm_plausible():
    box, scale, geo = np.array([80, 60, 160, 300], np.float32), 30.0, GeometryConfig()
    assert arm_plausible(_kpts(), box, scale, geo) == (True, True)
    assert arm_plausible(_kpts(l_elb=(100, 250)), box, scale, geo) == (False, True)  # плечо 5*scale
    assert arm_plausible(_kpts(l_wri=(10, 130)), box, scale, geo) == (False, True)  # предплечье 3*scale и за рамкой
    assert arm_plausible(_kpts(l_wri=(40, 130)), box, scale, geo) == (False, True)  # запястье за рамкой > scale
    off = dataclasses.replace(geo, max_upper_arm=0, max_forearm=0, max_wrist_outside=0)
    assert arm_plausible(_kpts(l_elb=(100, 250)), box, scale, off) == (True, True)


def test_event_box_median_resists_outlier():
    sig = make_signal(puffs([3]))
    k = sig.index_at(4.0)
    sig.boxes[k] = [0, 0, 400, 400]  # в пике рамка на двоих (перекрытие)
    box = event_box(sig, 4.0, EpisodeConfig())
    assert np.allclose(box, [100, 100, 160, 300])
    assert np.allclose(event_box(sig, 4.0, EpisodeConfig(box_window=0)), [0, 0, 400, 400])
