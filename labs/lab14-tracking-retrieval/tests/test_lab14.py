"""Offline behavior checks; CVCOURSE_LAB_NB selects a filled notebook.

Only explicit unfilled TODO functions skip; import/definition failures are errors.
No reference implementation is embedded in this public-compatible test module.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
DEFAULT = LAB/('lab14_solution.ipynb' if (LAB/'lab14_solution.ipynb').is_file() else 'lab14.ipynb')
NB = Path(os.environ.get('CVCOURSE_LAB_NB', DEFAULT)).resolve()


@pytest.fixture(scope='module')
def sol():
    ns = dict(LAB_DIR=NB.parent, __name__='lab14_tests', __todo__=set())
    for cell in json.loads(NB.read_text(encoding='utf-8'))['cells']:
        if 'lab14-definitions' not in cell.get('metadata', {}).get('tags', []):
            continue
        source = ''.join(cell['source'])
        exec(compile(source, str(NB), 'exec'), ns)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and any(
                    isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                    and isinstance(n.exc.func, ast.Name) and n.exc.func.id == 'NotImplementedError'
                    for n in ast.walk(node)):
                ns['__todo__'].add(node.name)
    for name in ('make_scene', 'array_hashes', 'iou', 'greedy_match', 'color_track', 'track_boxes',
                 'identity_report', 'build_bow', 'query_vector', 'verify_pair', 'search'):
        assert callable(ns.get(name)), 'Missing definition: '+name
    return ns


def need(sol, *names):
    unfinished = sol['__todo__'].intersection(names)
    if unfinished:
        pytest.skip('Unfilled TODO: '+', '.join(sorted(unfinished)))


@pytest.fixture(scope='module')
def scene(sol):
    return sol['make_scene']()


@pytest.fixture(scope='module')
def bow(sol, scene):
    need(sol, 'build_bow')
    return sol['build_bow'](scene['train_desc'], [scene[f'db{i}_desc'] for i in range(7)])


def test_environment(sol):
    assert cv2.__version__ == '4.14.0'
    assert importlib.metadata.version('opencv-contrib-python') == '4.14.0.94'


def test_data_hashes_and_export(sol, scene, tmp_path):
    readme = (LAB/'data/README.md').read_text(encoding='utf-8')
    hashes = sol['array_hashes'](scene)
    assert len(hashes) >= 30
    for key, value in hashes.items():
        assert f'| {key} |' in readme and value in readme
    sol['data_module'].export(tmp_path, scene)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    sol['data_module'].export(tmp_path, sol['make_scene']())
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    manifest = json.loads(first['manifest.json'])
    assert manifest['archive_sha256'] == hashlib.sha256(first['scene.npz']).hexdigest()
    assert manifest['archive_sha256'] in readme
    with np.load(tmp_path/'scene.npz', allow_pickle=False) as archive:
        assert sol['array_hashes']({k: archive[k] for k in archive.files}) == hashes


def test_separate_training_images_and_fresh_seed(sol, scene):
    assert scene['train_images'].shape == (6, 224, 224)
    for train in scene['train_images']:
        assert not any(np.array_equal(train, db) for db in scene['database'])
    other = sol['make_scene'](seed=17)
    assert not np.array_equal(other['train_desc'], scene['train_desc'])
    assert not np.array_equal(other['database'], scene['database'])
    np.testing.assert_array_equal(other['boxes_gt'], scene['boxes_gt'])


def test_ready_descriptors_and_scrambled_layout(sol, scene):
    assert scene['train_desc'].shape[1] == 128 and scene['train_desc'].dtype == np.float32
    assert scene['blank_desc'].shape == (0, 128)
    assert len(scene['query_xy']) == len(scene['query_desc'])
    original, scrambled = scene['database'][[0, 6]]
    for j, source in enumerate(scene['permutation']):
        y, x = divmod(j, 4)
        sy, sx = divmod(int(source), 4)
        np.testing.assert_array_equal(scrambled[y*56:(y+1)*56, x*56:(x+1)*56],
                                      original[sy*56:(sy+1)*56, sx*56:(sx+1)*56])


def test_iou_analytic(sol):
    assert sol['iou']([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1/3)
    assert sol['iou']([0, 0, 10, 10], [10, 0, 10, 10]) == 0
    assert sol['iou']([0, 0, 10, 10], [0, 0, 10, 10]) == 1


@pytest.mark.parametrize('a,b', [([], []), ([], [[0, 0, 10, 10]]), ([[0, 0, 10, 10]], [])])
def test_association_empty(sol, a, b):
    need(sol, 'greedy_match')
    assert sol['greedy_match'](a, b) == []


def test_association_unique_tie_order_and_threshold(sol):
    need(sol, 'greedy_match')
    a = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert sol['greedy_match'](a, a) == [(0, 0), (1, 1)]
    assert sol['greedy_match'](a, [[8, 0, 10, 10]], .2) == []
    assert sol['greedy_match'](a, [[0, 0, 10, 10]], 1.) == [(0, 0)]
    assert sol['greedy_match']([[50, 0, 10, 10], [0, 0, 10, 10]],
                               [[0, 0, 10, 10], [50, 0, 10, 10]]) == [(0, 1), (1, 0)]


@pytest.mark.parametrize('kwargs', [dict(threshold=0), dict(threshold=1.1), dict(threshold=float('nan'))])
def test_association_bad_threshold(sol, kwargs):
    need(sol, 'greedy_match')
    with pytest.raises(ValueError):
        sol['greedy_match']([], [], **kwargs)


def test_association_bad_boxes(sol):
    need(sol, 'greedy_match')
    for box in ([0, 0, -1, 2], [0, 0, float('nan'), 2]):
        with pytest.raises(ValueError):
            sol['greedy_match']([box], [])


def test_color_scale_adaptation(sol, scene):
    rows = {}
    for name in ('MeanShift', 'CamShift'):
        boxes = sol['color_track'](scene['color_frames'], scene['color_gt'][0], name)
        rows[name] = np.mean([sol['iou'](a, b) for a, b in zip(boxes, scene['color_gt'])])
    assert rows['MeanShift'] == pytest.approx(.5744748524, abs=.015)
    assert rows['CamShift'] > .92 and rows['CamShift'] > rows['MeanShift']+.2


def test_color_input_validation(sol, scene):
    with pytest.raises(ValueError):
        sol['color_track'](scene['color_frames'], scene['color_gt'][0], 'typo')


def test_tracking_crossing_and_missing_observations(sol, scene):
    need(sol, 'greedy_match')
    result = sol['track_boxes'](scene['frames'])
    row = sol['identity_report'](result, scene['boxes_gt'], scene['visible'])
    assert row['id_switches'] == 0 and row['created_ids'] == 2
    assert row['coverage'] > .8 and row['eligible'] == 78 and row['center_error_px'] < 1
    assert result['rejected_merges'] > 0
    assert all(not r['observed'] for r in result['history'][20])
    assert any(not r['observed'] and r['missed'] == 2 for r in result['history'][8])
    assert {r['id'] for r in result['history'][-1]} == {0, 1}
    baseline = sol['identity_report'](sol['track_boxes'](scene['frames'], predict=False),
                                      scene['boxes_gt'], scene['visible'])
    assert baseline['id_switches'] > row['id_switches']


def test_tracking_expiration_and_new_identity(sol, scene):
    need(sol, 'greedy_match')
    frames = np.concatenate([scene['frames'][:6], np.full_like(scene['frames'][:10], 50), scene['frames'][:1]])
    result = sol['track_boxes'](frames, max_missed=3)
    assert result['history'][9] == []
    assert result['created_ids'] == 4
    assert {r['id'] for r in result['history'][-1]} == {2, 3}
    assert sol['track_boxes']([])['history'] == []


def test_identity_evaluation_does_not_count_forecasts(sol):
    need(sol, 'greedy_match')
    result = dict(history=[[dict(id=0, box=[0, 0, 10, 10], observed=False)]], created_ids=1, rejected_merges=0)
    row = sol['identity_report'](result, np.array([[[0, 0, 10, 10]]]), [[True]])
    assert row['coverage'] == 0 and row['matched'] == 0 and row['center_error_px'] is None


def test_bow_counts_and_idf_database_only(sol, scene, bow):
    counts = bow['counts']
    assert counts.shape == (7, 200)
    np.testing.assert_array_equal(counts.sum(1), [len(scene[f'db{i}_desc']) for i in range(7)])
    np.testing.assert_allclose(bow['idf'], np.log(8/(np.count_nonzero(counts, axis=0)+1))+1)
    np.testing.assert_allclose(np.linalg.norm(bow['vectors'], axis=1), 1, atol=1e-12)
    assert bow['training_count'] == len(scene['train_desc'])


def test_query_uses_frozen_idf_and_norm(sol, scene, bow):
    centers, idf = bow['model'].cluster_centers_.copy(), bow['idf'].copy()
    q = sol['query_vector'](scene['query_desc'], bow)
    assert np.linalg.norm(q) == pytest.approx(1)
    np.testing.assert_allclose(q, sol['query_vector'](np.tile(scene['query_desc'], (3, 1)), bow), atol=1e-12)
    assert not sol['query_vector'](scene['blank_desc'], bow).any()
    np.testing.assert_array_equal(centers, bow['model'].cluster_centers_)
    np.testing.assert_array_equal(idf, bow['idf'])


def test_vocabulary_uses_training_corpus_and_zero_document(sol):
    need(sol, 'build_bow')
    train = np.float32([[0, 0], [.1, 0], [10, 10], [10.1, 10]])
    bow = sol['build_bow'](train, [np.float32([[1000, 1000]]), np.empty((0, 2), np.float32)], k=2)
    assert bow['model'].cluster_centers_.max() < 11
    assert not bow['vectors'][1].any()
    assert np.isfinite(bow['idf']).all()
    np.testing.assert_array_equal(bow['counts'].sum(1), [1, 0])


@pytest.mark.parametrize('k', [0, 1, 5])
def test_bow_invalid_word_count(sol, k):
    need(sol, 'build_bow')
    with pytest.raises(ValueError):
        sol['build_bow'](np.zeros((4, 2), np.float32), [np.zeros((1, 2), np.float32)], k)


def test_retrieval_geometry_fixes_scrambled_first_hit(sol, scene, bow):
    need(sol, 'verify_pair')
    ds = [scene[f'db{i}_desc'] for i in range(7)]
    xy = [scene[f'db{i}_xy'] for i in range(7)]
    row = sol['search'](scene['query_desc'], scene['query_xy'], scene['query'].shape,
                        ds, xy, [im.shape for im in scene['database']], bow)
    assert row['ranking'][0] == 6 and row['accepted_ids'] == [0]
    checked = {r['id']: r for r in row['checked']}
    assert checked[0]['fraction'] > .9 and checked[0]['inliers'] >= 170
    assert checked[0]['reprojection_px'] < .5 and min(checked[0]['coverage'], checked[0]['candidate_coverage']) > .5
    assert not checked[6]['accepted'] and checked[6]['fraction'] < .2
    # Direction check against independent scene H, on a grid rather than fitted matches.
    grid = np.float32([[[40, 40]], [[180, 40]], [[180, 180]], [[40, 180]]])
    expected = cv2.perspectiveTransform(grid, np.linalg.inv(scene['query_H']))
    actual = cv2.perspectiveTransform(grid, checked[0]['H'])
    assert np.max(np.linalg.norm(actual-expected, axis=-1)) < .6


@pytest.mark.parametrize('name', ['foreign', 'blank'])
def test_negative_queries(sol, scene, bow, name):
    need(sol, 'verify_pair')
    result = sol['search'](scene[name+'_desc'], scene[name+'_xy'], scene[name].shape,
                           [scene[f'db{i}_desc'] for i in range(7)], [scene[f'db{i}_xy'] for i in range(7)],
                           [im.shape for im in scene['database']], bow)
    assert not result['accepted_ids']
    if name == 'blank':
        assert result['ranking'] == [0, 1, 2, 3, 4] and not any(result['scores'])


@pytest.mark.parametrize('case', ['collinear', 'tiny_support', 'invalid_projection'])
def test_geometry_rejects_local_or_degenerate_agreement(sol, case):
    points = np.float32([[x, y] for x in (20, 60, 100, 140) for y in (20, 60, 100, 140)])
    H = np.eye(3)
    if case == 'collinear': points[:, 1] = 30
    if case == 'tiny_support': points = points*.02+20
    if case == 'invalid_projection': H = np.diag([.01, .01, 1.])
    dst = cv2.perspectiveTransform(points[:, None], H)[:, 0]
    row = sol['geometry_report'](points, dst, H, np.ones(16, bool), (224, 224), (224, 224))
    assert not row['accepted']


def test_ransac_exact_and_no_model_cases(sol):
    need(sol, 'verify_pair')
    rng = np.random.default_rng(71)
    xy = rng.uniform(10, 210, (50, 2)).astype(np.float32)
    desc = rng.uniform(0, 100, (50, 128)).astype(np.float32)
    dst = xy+[3, 5]
    row = sol['verify_pair'](xy, desc, dst.astype(np.float32), desc.copy(), (224, 224), (224, 224))
    assert row['accepted'] and row['inliers'] == 50 and row['reprojection_px'] < 1e-3
    row = sol['verify_pair'](xy[:3], desc[:3], xy, desc, (224, 224), (224, 224))
    assert not row['accepted'] and row['H'] is None
    line = np.column_stack([np.arange(50), np.ones(50)]).astype(np.float32)
    assert not sol['verify_pair'](line, desc, line, desc, (224, 224), (224, 224))['accepted']


def test_geometry_new_scene_seed(sol):
    need(sol, 'verify_pair')
    sample = sol['make_scene'](seed=17)
    rows = [sol['verify_pair'](sample['query_xy'], sample['query_desc'], sample[f'db{i}_xy'],
                              sample[f'db{i}_desc'], sample['query'].shape, (224, 224)) for i in (0, 6)]
    assert rows[0]['accepted'] and not rows[1]['accepted']


def test_classical_dependencies_and_three_stubs():
    source = '\n'.join(''.join(c['source']) for c in json.loads(NB.read_text(encoding='utf-8'))['cells']
                       if c['cell_type'] == 'code')
    tree = ast.parse(source)
    imports = {n.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    assert imports <= {'importlib', 'json', 'pathlib', 'cv2', 'numpy', 'sklearn', 'IPython'}
    assert all(not (isinstance(n, ast.Attribute) and n.attr == 'dnn') for n in ast.walk(tree))
    if '_solution' in NB.name:
        assert source.count('# >>> SOL TODO ') == 3
    else:
        assert source.count('raise NotImplementedError(') == 3
