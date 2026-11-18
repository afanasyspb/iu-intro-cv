"""Autonomous offline checks for the starter and the TA reference.

Select another notebook with CVCOURSE_LAB_NB. Only a function containing an
explicit NotImplementedError is skipped; missing definitions and other errors fail.
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
DEFAULT = HERE / ('lab12_solution.ipynb' if (HERE / 'lab12_solution.ipynb').is_file() else 'lab12.ipynb')
NB = Path(os.environ.get('CVCOURSE_LAB_NB', DEFAULT)).resolve()
HASHES = {
    'left': 'd82652c48a0e8ec160e22dfac1b9161ac8ef7cb65dceb2d0cab7984d9ca8302e',
    'right': '70738ec4baa987d79990aad0d6d25ae13319fd713e6c03360af986bbbd900459',
    'gt': '3fa843c9acfbf46a59c96f8d29e1d8f3fc62b4a8c01aea3f2737165a3832e2ba',
    'valid': 'fb7f225ddc8667a1c29cd9e100ebbd9daec6188a21c3672213ab9b4f53090128',
    'depth_gt': 'edc3adb28d976feaef3c79d39b3f67d79f0e601cb918fc17e50eb1e6d0c16a14',
}


@pytest.fixture(scope='module')
def sol():
    notebook = json.loads(NB.read_text(encoding='utf-8'))
    namespace = {'__name__': 'lab12_tests', 'LAB_DIR': NB.parent, '__todo__': set()}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or 'lab12-definitions' not in cell.get('metadata', {}).get('tags', []):
            continue
        source = ''.join(cell['source'])
        exec(compile(source, str(NB), 'exec'), namespace)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name) and n.exc.func.id == 'NotImplementedError'
                for n in ast.walk(node)
            ):
                namespace['__todo__'].add(node.name)
    for name in ('make_scene', 'array_hashes', 'rectify_pair', 'sgbm_wls', 'disparity_report',
                 'fundamental', 'epipolar_distances', 'stereo_depth', 'cloud_from_disparity', 'write_ply'):
        assert callable(namespace.get(name)), f'Missing definition: {name}'
    return namespace


def need(sol, *names):
    todo = sol['__todo__'].intersection(names)
    if todo:
        pytest.skip('Unfilled TODO: ' + ', '.join(sorted(todo)))


@pytest.fixture(scope='module')
def scene(sol):
    return sol['make_scene']()


@pytest.fixture(scope='module')
def rect(sol, scene):
    need(sol, 'rectify_pair')
    return sol['rectify_pair'](scene['left'], scene['right'], scene['K'], scene['dist'],
                               scene['K'], scene['dist'], scene['R'], scene['T'])


@pytest.fixture(scope='module')
def matched(sol, rect):
    need(sol, 'sgbm_wls')
    return sol['sgbm_wls'](rect['left'], rect['right'])


def test_environment_and_definitions(sol):
    assert cv2.__version__ == '4.14.0'
    assert importlib.metadata.version('opencv-contrib-python') == '4.14.0.94'


def test_dataset_hashes_and_repeatability(sol, scene):
    assert sol['array_hashes'](scene) == HASHES
    assert sol['array_hashes'](sol['make_scene']()) == HASHES
    other = sol['make_scene'](seed=71)
    assert not np.array_equal(other['left'], scene['left'])
    np.testing.assert_array_equal(other['gt'], scene['gt'])


def test_ground_truth_world_geometry(scene):
    assert scene['size'] == (384, 240) and scene['left'].shape == scene['gt'].shape == (240, 384)
    assert scene['left'].dtype == np.uint8 and scene['valid'].dtype == bool
    assert scene['valid'].sum() > 50000
    np.testing.assert_allclose(-scene['R'].T @ scene['T'], [.12, 0, 0], atol=1e-12)
    assert set(np.round(scene['points'][:, 2], 8)) == {2., 4.}
    # Triangulate known pixel correspondences using independent projection matrices.
    P1 = scene['K'] @ np.column_stack([np.eye(3), np.zeros(3)])
    P2 = scene['K'] @ np.column_stack([scene['R'], scene['T']])
    homogeneous = cv2.triangulatePoints(P1, P2, scene['pts1'].T, scene['pts2'].T)
    np.testing.assert_allclose((homogeneous[:3] / homogeneous[3]).T, scene['points'], atol=1e-10)


def test_epipolar_constraint_and_perturbed_correspondence(sol, scene):
    F = sol['fundamental'](scene['K'], scene['K'], scene['R'], scene['T'])
    assert np.linalg.matrix_rank(F) == 2
    assert sol['epipolar_distances'](scene['pts1'], scene['pts2'], F).max() < 1e-8
    assert np.median(sol['epipolar_distances'](scene['pts1'], scene['pts2'] + [0, 3], F)) > 2.9


def test_rectification_rows_projection_and_pixels(sol, scene, rect):
    corrected = []
    for name, rotation, projection in (('pts1', 'R1', 'P1'), ('pts2', 'R2', 'P2')):
        corrected.append(cv2.undistortPoints(scene[name].reshape(-1, 1, 2), scene['K'], scene['dist'],
                                            R=rect[rotation], P=rect[projection]).reshape(-1, 2))
    assert np.median(np.abs(scene['pts1'][:, 1] - scene['pts2'][:, 1])) > 1
    assert np.max(np.abs(corrected[0][:, 1] - corrected[1][:, 1])) < 1e-8
    assert rect['left'].dtype == rect['right'].dtype == np.uint8
    assert not np.array_equal(rect['left'], scene['left'])
    assert rect['P2'][0, 3] < 0 and rect['Q'][3, 2] > 0
    assert abs(-rect['P2'][0, 3] / rect['P2'][0, 0] - .12) < 1e-12


def test_rectification_independent_camera_geometry(sol):
    need(sol, 'rectify_pair')
    K = np.float64([[200, 0, 90], [0, 210, 60], [0, 0, 1]])
    dist = np.float64([-.1, .01, .001, 0, 0])
    R = cv2.Rodrigues(np.float64([-.025, .04, .01]))[0]
    T = np.float64([-.2, .003, .002])
    image = np.zeros((120, 180), np.uint8)
    result = sol['rectify_pair'](image, image, K, dist, K, dist, R, T)
    points = np.float64([[-.4, -.2, 3], [.3, .1, 2], [.1, -.3, 5], [-.2, .2, 4]])
    p1 = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)[0]
    p2 = cv2.projectPoints(points, cv2.Rodrigues(R)[0], T, K, dist)[0]
    a = cv2.undistortPoints(p1, K, dist, R=result['R1'], P=result['P1'])[:, 0]
    b = cv2.undistortPoints(p2, K, dist, R=result['R2'], P=result['P2'])[:, 0]
    np.testing.assert_allclose(a[:, 1], b[:, 1], atol=1e-7)


def test_rectification_bad_inputs(sol, scene):
    need(sol, 'rectify_pair')
    for left, right, T in ((scene['left'].astype(float), scene['right'], scene['T']),
                            (scene['left'], scene['right'][:20], scene['T']),
                            (scene['left'], scene['right'], np.zeros(3))):
        with pytest.raises(ValueError):
            sol['rectify_pair'](left, right, scene['K'], scene['dist'], scene['K'], scene['dist'], scene['R'], T)


def test_sgbm_fixed_point_units_and_right_sign(matched):
    assert matched['raw_left'].dtype == matched['raw_right'].dtype == np.int16
    good = matched['raw_left'] > 0
    np.testing.assert_array_equal(matched['disparity'][good], matched['raw_left'][good] / 16)
    assert np.isnan(matched['disparity'][~good]).all()
    assert np.median(matched['raw_right'][matched['raw_right'] < 0]) < -16
    assert np.nanmedian(matched['wls']) < 40
    assert np.isfinite(matched['confidence']).all() and matched['confidence'].min() >= 0
    assert matched['confidence'].max() <= 255
    assert np.nanmean(np.abs(matched['wls'] - matched['disparity'])) > 0


def test_sgbm_and_wls_against_gt(sol, scene, matched):
    need(sol, 'disparity_report')
    for name in ('disparity', 'wls'):
        report = sol['disparity_report'](matched[name], scene['gt'], scene['valid'])
        assert report['coverage_pct'] > 95 and report['bad2_pct'] < 10
        assert report['rmse_valid_px'] < 1.5


def test_sgbm_independent_seed_and_disparity(sol):
    need(sol, 'sgbm_wls')
    rng = np.random.default_rng(991)
    left = rng.integers(20, 236, (96, 240), dtype=np.uint8)
    right = rng.integers(20, 236, left.shape, dtype=np.uint8)
    right[:, :-13] = left[:, 13:]
    result = sol['sgbm_wls'](left, right, 32, 3, 5, 0)
    assert abs(np.nanmedian(result['disparity'][8:-8, 70:-20]) - 13) < .1
    assert abs(np.nanmedian(result['wls'][8:-8, 70:-20]) - 13) < .1


@pytest.mark.parametrize('kwargs', [{'num_disparities': 15}, {'num_disparities': 0},
                                    {'block_size': 4}, {'uniqueness': -1}, {'speckle': -1}])
def test_sgbm_invalid_parameters(sol, kwargs):
    need(sol, 'sgbm_wls')
    with pytest.raises(ValueError):
        sol['sgbm_wls'](np.zeros((80, 200), np.uint8), np.zeros((80, 200), np.uint8), **kwargs)


def test_bad2_strict_boundary_and_mask(sol):
    need(sol, 'disparity_report')
    report = sol['disparity_report'](np.array([8., 10., 11., 999.]), np.full(4, 8.), [1, 1, 1, 0])
    assert report['valid_pixels'] == report['predicted_pixels'] == 3
    assert report['coverage_pct'] == 100
    assert abs(report['bad2_pct'] - 100 / 3) < 1e-12
    assert abs(report['rmse_px'] - np.sqrt(13 / 3)) < 1e-12


def test_missing_predictions_are_errors(sol):
    need(sol, 'disparity_report')
    result = sol['disparity_report'](np.array([np.nan, np.inf, 0., -1., 8.]), np.full(5, 8.), np.ones(5, bool))
    assert result['bad2_pct'] == 80 and result['coverage_pct'] == 20
    assert result['rmse_px'] is None and result['rmse_valid_px'] == 0
    result = sol['disparity_report'](np.array([np.nan]), np.array([8.]), [True])
    assert result['bad2_pct'] == 100 and result['coverage_pct'] == 0
    assert result['rmse_px'] is result['rmse_valid_px'] is None


def test_metrics_reject_empty_and_wrong_shapes(sol):
    need(sol, 'disparity_report')
    for pred, gt, mask in (([8.], [0.], [True]), ([8.], [np.nan], [True]),
                            ([8.], [8.], [False]), ([8., 8.], [8.], [True])):
        with pytest.raises(ValueError):
            sol['disparity_report'](pred, gt, mask)


def test_depth_units_offset_and_invalid_disparity(sol):
    z = sol['stereo_depth'](np.array([8., 16., 0., -1., np.nan, np.inf]), 520, .12, 2)
    np.testing.assert_allclose(z[:2], [6.24, 62.4 / 18])
    assert np.isnan(z[2:]).all()
    for focal, baseline in ((0, .1), (200, -1), (np.nan, .1)):
        with pytest.raises(ValueError):
            sol['stereo_depth'](np.array([8.]), focal, baseline)


def test_q_reconstruction_matches_independent_world_and_gt(sol, scene, rect):
    xyz = sol['cloud_from_disparity'](scene['gt'], rect['Q'])
    np.testing.assert_allclose(xyz[..., 2], scene['depth_gt'], rtol=2e-6)
    v, u = np.indices(scene['gt'].shape)
    K = rect['P1'][:, :3]
    np.testing.assert_allclose(xyz[..., 0] / xyz[..., 2] * K[0, 0] + K[0, 2], u, atol=5e-5)
    np.testing.assert_allclose(xyz[..., 1] / xyz[..., 2] * K[1, 1] + K[1, 2], v, atol=5e-5)
    # Rotate reconstruction back to the original left frame: both physical planes stay at Z=2/4.
    world = xyz @ rect['R1']
    assert np.max(np.minimum(abs(world[..., 2] - 2), abs(world[..., 2] - 4))) < 2e-6


def test_ply_header_vertices_colors_and_invalid_filter(sol, tmp_path):
    xyz = np.float32([[[1, 2, 3], [np.nan, 2, 3]], [[1, 2, -3], [4, 5, 6]]])
    image = np.uint8([[10, 20], [30, 40]])
    output = tmp_path / 'cloud.ply'
    assert sol['write_ply'](output, xyz, image) == 2
    content = output.read_text(encoding='utf-8')
    assert 'element vertex 2\n' in content and 'property uchar blue\n' in content
    data = np.loadtxt(content.split('end_header\n')[1].splitlines())
    np.testing.assert_allclose(data, [[1, 2, 3, 10, 10, 10], [4, 5, 6, 40, 40, 40]])
    mask = np.array([[0, 1], [1, 1]], bool)
    assert sol['write_ply'](output, xyz, image, mask) == 1


def test_depth_sensitivity_local_derivative(sol):
    z = np.array([1., 2., 4., 8.])
    f, B, eps = 400., .12, 1e-4
    d = f * B / z
    exact = abs(sol['stereo_depth'](d + eps, f, B) - sol['stereo_depth'](d - eps, f, B)) / (2 * eps)
    np.testing.assert_allclose(exact, z**2 / (f * B), rtol=1e-8)
