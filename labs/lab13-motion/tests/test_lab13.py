"""Offline numerical checks. Select a notebook using CVCOURSE_LAB_NB.

Only functions with an explicit unfilled NotImplementedError are skipped.
Definition/import errors and failed numerical checks remain failures.
"""
import ast
import importlib.metadata
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
DEFAULT = HERE / ('lab13_solution.ipynb' if (HERE / 'lab13_solution.ipynb').is_file() else 'lab13.ipynb')
NB = Path(os.environ.get('CVCOURSE_LAB_NB', DEFAULT)).resolve()
HASHES = {
    'a': '6acd93ed45589049131b9d54976be2ea99f6d5ef7e0b95820ceb93e9793dc062',
    'b': '0ba11297f96957b8e4a46e6e7f9dad590b9544c7d66d6f07cd4e288bf3ad9989',
    'edge': '1633211a3b4feb4fd581809ea9405f06ccc19418437a3fe15ea1f941a835fb73',
    'gt': 'dce7690f439ec65c674545081db0c57ac1079b1236ba48334c117761f5ce8999',
    'occluded': 'a0d1c796ac4ca093d397933603903d7d450662778b58ead5965684a1d5552963',
    'shadow_gt': '9473393a48f5bc97cccee19dd8baff28f5fadd8e28c2fa1b3a8d4b9b55faebd9',
    'shift': '0b17d0424fc8c1857c2ab37732495673a92e0f47d5ba491baa2798d7dc058279',
    'valid': '9c3e681737b484706f34fbbdf77ad709f930117693620f5f8704a0878446864d',
    'video': '3c3572037de2e0f49461f5dad2c19195f19b1596242f40913acc229b00c7487a',
    'video_gt': 'cb89e1dd3f44c538cb86c59804a9ed1e6641b488c01613e76a5d2e850e3513cc',
    'visible': '8c420b0627cd95b37ed4b942dea1cfb08d68034ac5a81af8d62b6dbc1b1d282f',
}


@pytest.fixture(scope='module')
def sol():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    ns = dict(LAB_DIR=NB.parent, __name__='lab13_tests', __todo__=set())
    for cell in nb['cells']:
        if cell['cell_type'] != 'code' or 'lab13-definitions' not in cell.get('metadata', {}).get('tags', []):
            continue
        source = ''.join(cell['source'])
        exec(compile(source, str(NB), 'exec'), ns)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name) and n.exc.func.id == 'NotImplementedError'
                for n in ast.walk(node)
            ):
                ns['__todo__'].add(node.name)
    for name in ('make_scene', 'array_hashes', 'mask_report', 'flow_report', 'background_masks',
                 'sparse_lk', 'dense_pair', 'track_report', 'clinic_disparity'):
        assert callable(ns.get(name)), 'Missing definition: ' + name
    return ns


def need(sol, *names):
    todo = sol['__todo__'].intersection(names)
    if todo:
        pytest.skip('Unfilled TODO: ' + ', '.join(sorted(todo)))


@pytest.fixture(scope='module')
def scene(sol):
    return sol['make_scene']()


def test_environment_and_definitions(sol):
    assert cv2.__version__ == '4.14.0'
    assert importlib.metadata.version('opencv-contrib-python') == '4.14.0.94'


def test_hashes_and_independent_seed(sol, scene):
    assert sol['array_hashes'](scene) == HASHES
    assert sol['array_hashes'](sol['make_scene']()) == HASHES
    other = sol['make_scene'](seed=71)
    assert not np.array_equal(other['a'], scene['a'])
    np.testing.assert_array_equal(other['gt'], scene['gt'])
    np.testing.assert_array_equal(other['valid'], scene['valid'])


def test_ground_truth_translation_and_visibility(scene):
    assert scene['a'].shape == (128, 176) and scene['a'].dtype == np.uint8
    np.testing.assert_array_equal(scene['b'][3:, 6:], scene['a'][:-3, :-6])
    assert scene['valid'].sum() == 10206
    assert scene['visible'].sum() < scene['valid'].sum()
    y, x = np.where(scene['visible'])
    assert not np.any((x+6 >= 70) & (x+6 < 114) & (y+3 >= 40) & (y+3 < 88))
    np.testing.assert_array_equal(scene['gt'][scene['valid']], np.tile([6., 3.], (10206, 1)))


def test_background_gt_warmup_stop_and_shadow(scene):
    assert scene['video'].shape == (80, 96, 144, 3)
    assert not scene['video_gt'][:24].any()
    assert scene['video_gt'][24:].sum(axis=(1, 2)).tolist() == [440] * 56
    np.testing.assert_array_equal(scene['video_gt'][46], scene['video_gt'][-1])
    assert not (scene['shadow_gt'] & scene['video_gt']).any()


def test_frame_difference_is_not_object_mask(sol, scene):
    a, b = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in scene['video'][-2:]]
    assert sol['mask_report'](cv2.absdiff(a, b) > 25, scene['video_gt'][-1])['iou'] < .05


def test_mask_metrics_analytic(sol):
    row = sol['mask_report']([True, True, False, False], [True, False, True, False])
    assert (row['tp'], row['fp'], row['fn']) == (1, 1, 1)
    assert row['iou'] == 1/3
    assert sol['mask_report'](np.zeros((2, 3)), np.zeros((2, 3)))['iou'] == 1
    with pytest.raises(ValueError):
        sol['mask_report']([1], [1, 0])


def test_epe_analytic_mask_and_missing_predictions(sol):
    gt = np.array([[3., 4.], [0., 0.], [6., 8.]])
    row = sol['flow_report'](np.zeros_like(gt), gt, [True, False, True])
    assert row == dict(epe=7.5, conditional_epe=7.5, coverage=1., gt_count=2)
    prediction = np.zeros_like(gt); prediction[0] = np.nan
    row = sol['flow_report'](prediction, gt, [True, False, True])
    assert row == dict(epe=None, conditional_epe=10., coverage=.5, gt_count=2)
    prediction[:] = np.nan
    assert sol['flow_report'](prediction, gt, [True, False, True])['conditional_epe'] is None


@pytest.mark.parametrize('case', ['empty', 'shape', 'invalid_gt'])
def test_epe_rejects_invalid_support(sol, case):
    gt = np.zeros((2, 3, 2)); mask = np.ones((2, 3), bool)
    prediction = gt.copy()
    if case == 'empty': mask[:] = False
    if case == 'shape': prediction = np.zeros((2, 3))
    if case == 'invalid_gt': gt[0, 0, 0] = np.nan
    with pytest.raises(ValueError): sol['flow_report'](prediction, gt, mask)


@pytest.mark.parametrize('kind', ['MOG2', 'KNN'])
def test_background_quality_shadows_and_morphology(sol, scene, kind):
    need(sol, 'background_masks')
    result = sol['background_masks'](scene['video'], kind)
    assert result['raw'].shape == result['clean'].shape == scene['video_gt'].shape
    assert set(np.unique(result['raw'])) <= {0, 127, 255}
    assert set(np.unique(result['clean'])) <= {0, 255}
    labels, clean, truth = result['raw'][38], result['clean'][38], scene['video_gt'][38]
    row = sol['mask_report'](clean == 255, truth)
    raw = sol['mask_report'](labels == 255, truth)
    assert row['iou'] > .75 and row['fp'] < 150
    assert row['fp'] < raw['fp'] and np.count_nonzero(labels == 127) > 100
    wrong = sol['mask_report'](labels > 0, truth)
    assert wrong['fp'] > raw['fp'] + 100


def test_background_frozen_vs_adapting(sol, scene):
    need(sol, 'background_masks')
    frozen = sol['background_masks'](scene['video'], adapt=0.)['clean'][-1]
    fast = sol['background_masks'](scene['video'], adapt=.1)['clean'][-1]
    assert sol['mask_report'](frozen == 255, scene['video_gt'][-1])['iou'] > .9
    assert sol['mask_report'](fast == 255, scene['video_gt'][-1])['iou'] < .1


@pytest.mark.parametrize('kwargs', [dict(kind='typo'), dict(adapt=-1), dict(history=0)])
def test_background_invalid_configuration(sol, scene, kwargs):
    need(sol, 'background_masks')
    with pytest.raises(ValueError): sol['background_masks'](scene['video'], **kwargs)


@pytest.mark.parametrize('seed,shift', [(13, (6, 3)), (71, (-4, 2))])
def test_sparse_translation_and_sign(sol, seed, shift):
    need(sol, 'sparse_lk')
    sample = sol['make_scene'](seed=seed, shift=shift)
    points = cv2.goodFeaturesToTrack(sample['a'], 80, .02, 10, mask=np.uint8(sample['valid'])*255)
    result = sol['sparse_lk'](sample['a'], sample['b'], points)
    row = sol['track_report'](result, sample['shift'])
    assert row['coverage'] > .9 and row['conditional_epe'] < .15
    assert np.all(result['status'][result['valid']])
    assert np.all(result['fb'][result['valid']] < 1.)


def test_pyramid_large_displacement_and_coverage(sol):
    need(sol, 'sparse_lk')
    sample = sol['make_scene'](shift=(18, -9))
    points = cv2.goodFeaturesToTrack(sample['a'],80,.02,10,mask=np.uint8(sample['valid'])*255)
    rows = [sol['track_report'](sol['sparse_lk'](sample['a'],sample['b'],points,max_level=level),sample['shift'])
            for level in (0,3)]
    assert rows[1]['coverage'] > .9 and rows[1]['conditional_epe'] < .15
    # Low conditional EPE can conceal rejected tracks; compare coverage too.
    assert rows[1]['coverage'] > rows[0]['coverage'] + .3


def test_occlusion_reverse_check_and_rejected_points(sol, scene):
    need(sol, 'sparse_lk')
    points = cv2.goodFeaturesToTrack(scene['a'],80,.02,10,mask=np.uint8(scene['valid'])*255).reshape(-1,2)
    result = sol['sparse_lk'](scene['a'],scene['occluded'],points)
    assert result['valid'].sum() < result['status'].sum()
    pixels = np.rint(points).astype(int)
    eligible = scene['visible'][pixels[:,1],pixels[:,0]]
    row = sol['track_report'](result,scene['shift'],eligible)
    assert row['coverage'] > .75 and row['conditional_epe'] < .2
    invalid = sol['sparse_lk'](scene['a'],scene['b'],np.float32([[-100,-100],[400,300]]))
    assert not invalid['valid'].any() and not invalid['status'].any()
    assert np.isnan(invalid['err']).all()


def test_sparse_flat_and_explicit_empty(sol, scene):
    need(sol, 'sparse_lk')
    flat = np.full_like(scene['a'], 100)
    for a, b, points in ((flat,flat,None),(scene['a'],scene['b'],np.empty((0,2)))):
        result = sol['sparse_lk'](a,b,points)
        assert result['q'].shape == (0,2) and result['valid'].size == 0


@pytest.mark.parametrize('seed,shift', [(13,(6,3)), (71,(-4,2))])
def test_dense_translation_and_fixed_support(sol, seed, shift):
    need(sol, 'dense_pair')
    sample = sol['make_scene'](seed=seed,shift=shift)
    result = sol['dense_pair'](sample['a'],sample['b'])
    assert result['flow'].shape == (*sample['a'].shape,2) and result['flow'].dtype == np.float32
    assert result['rgb'].shape == (*sample['a'].shape,3) and result['rgb'].dtype == np.uint8
    row = sol['flow_report'](result['flow'],sample['gt'],sample['valid'])
    assert row['coverage'] == 1. and row['epe'] < .4
    np.testing.assert_allclose(np.median(result['flow'][sample['valid']],axis=0),shift,atol=.2)


def test_hsv_fixed_scale_direction_and_zero(sol):
    directions = np.float32([[[10,0],[0,10],[-10,0],[0,-10],[0,0]]])
    rgb = sol['flow_color'](directions,10.)
    assert rgb[0,0,0] == 255 and rgb[0,0,1] == 0 and rgb[0,0,2] == 0
    assert np.max(np.abs(rgb[0,2].astype(int)-[0,255,255])) <= 1
    assert not rgb[0,4].any()
    np.testing.assert_array_equal(sol['flow_color'](directions[:,:1],20.)[0,0], [127,0,0])


def test_dense_zero_pair(sol, scene):
    need(sol, 'dense_pair')
    result = sol['dense_pair'](scene['a'],scene['a'])
    assert np.median(np.linalg.norm(result['flow'][scene['valid']],axis=1)) < .01


@pytest.mark.parametrize('function', ['sparse_lk','dense_pair'])
def test_flow_input_validation(sol, scene, function):
    need(sol, function)
    with pytest.raises(ValueError): sol[function](scene['a'].astype(float),scene['b'])
    with pytest.raises(ValueError): sol[function](scene['a'][:-1],scene['b'])


def test_aperture_degeneracy(sol, scene):
    edge = scene['edge'].astype(np.float32)
    gx = cv2.Sobel(edge,cv2.CV_32F,1,0,scale=1/8)
    gy = cv2.Sobel(edge,cv2.CV_32F,0,1,scale=1/8)
    A = np.column_stack([gx[20:44,34:46].ravel(),gy[20:44,34:46].ravel()])
    assert np.linalg.eigvalsh(A.T@A)[0] == 0
    x = sol['data_module'].translate(scene['edge'],(3,0))
    xy = sol['data_module'].translate(scene['edge'],(3,5))
    np.testing.assert_array_equal(x[12:-12,12:-12],xy[12:-12,12:-12])


def test_hw4_bad2_boundary_coverage_and_rms(sol):
    row = sol['clinic_disparity'](np.array([8.,10.,11.,np.nan]),np.full(4,8.),np.ones(4,bool))
    assert row['bad2'] == 50. and row['coverage'] == 75. and row['rmse'] is None
    assert abs(row['conditional_rmse'] - np.sqrt(13/3)) < 1e-12
    full = sol['clinic_disparity'](np.array([8.,10.]),np.full(2,8.),np.ones(2,bool))
    assert full['bad2'] == 0. and abs(full['rmse']-np.sqrt(2)) < 1e-12


def test_track_metric_denominator(sol):
    result = dict(points=np.zeros((3,2)),q=np.array([[3.,4.],[9.,9.],[3.,4.]]),valid=np.array([True,False,True]))
    row = sol['track_report'](result,np.zeros(2))
    assert row == dict(eligible=3,kept=2,coverage=2/3,conditional_epe=5.)
