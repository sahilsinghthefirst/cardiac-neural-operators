                                                                                 

                                                                            
                                                                             
                                                                          
                                                                          
   
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sufficient(truth, pred, valid):
    ok = valid & np.isfinite(truth) & np.isfinite(pred)
    return np.stack([ok.sum((1, 2)), np.where(ok, truth, 0).sum((1, 2)),
                     np.where(ok, truth * truth, 0).sum((1, 2)),
                     np.where(ok, (truth - pred) ** 2, 0).sum((1, 2))], axis=1)


def r2(stats):
    n, total, square, error = np.asarray(stats).sum(axis=-2).T
    return 1 - error / (square - total ** 2 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True, help='Root of the original benchmark checkout')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--coordinates', action='store_true')
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite revision-control outputs')
    out.mkdir(parents=True)
    analysis = root / 'outputs/icbinb_required_analysis'
    index_path = root / 'outputs/dataset_v1e_500_combined_pilot/deeponet_ready/sample_index.csv'
    index = pd.read_csv(index_path)
    train = index[index.split.eq('train')].drop_duplicates('base_sample_id').sort_values('base_sample_id')
    cohort = pd.read_csv(analysis / 'paired_test_cohort.csv').sort_values('cohort_index').reset_index(drop=True)
    assert len(train) == 350 and len(cohort) == 75
    assert not set(train.base_sample_id) & set(cohort.base_sample_id)
    maps = np.load(analysis / 'paired_test_cohort_maps.npz')
    valid = maps['valid_mask'].astype(bool)
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, 75, (5000, 75))
    train_ids = train.base_sample_id.tolist()
    records, means, per_case, sources = [], {}, {}, []
    for field, column in [('lat', 'LAT_grid_file'), ('apd90', 'APD90_grid_file')]:
        values, masks = [], []
        for row in train.itertuples():
            path = root / getattr(row, column)
            maskpath = root / row.valid_mask_grid_file
            value = np.load(path).astype(np.float64)
            mask = np.load(maskpath).astype(bool) & np.isfinite(value)
            values.append(value)
            masks.append(mask)
            sources.append({'path':str(path.relative_to(root)), 'sha256':sha(path)})
        values, masks = np.stack(values), np.stack(masks)
        counts = masks.sum(0)
        assert counts.min() > 0
        mean = np.where(masks, values, 0).sum(0) / counts
        means[field] = mean
        truth = maps['true_' + field].astype(np.float64)
        for model in ['training_mean', 'fno', 'deeponet']:
            pred = np.broadcast_to(mean, truth.shape) if model == 'training_mean' else maps[model+'_field_pred_'+field].astype(np.float64)
            stats = sufficient(truth, pred, valid)
            per_case[model+'_'+field] = stats
            for subset in ['all', 'conduction_only']:
                ids = np.arange(75) if subset == 'all' else np.flatnonzero(cohort.condition.eq('conduction_slowing_perturbation'))
                assert len(ids) == (75 if subset == 'all' else 18)
                bootdraws = draws if subset == 'all' else ids[np.random.default_rng(20261009).integers(0,len(ids),(5000,len(ids)))]
                boots = r2(stats[bootdraws])
                ci = np.percentile(boots,[2.5,97.5])
                sse = stats[ids,3].sum()
                records.append({'model':model,'field':field,'subset':subset,'n':len(ids),'r2':float(r2(stats[ids])),
                                'ci_lower':float(ci[0]),'ci_upper':float(ci[1]),'sse':float(sse)})
    table = pd.DataFrame(records)
    table.to_csv(out/'training_mean_metrics.csv',index=False)
    np.savez_compressed(out/'training_mean_maps.npz',**means)
    pd.DataFrame({'base_sample_id':train_ids}).to_csv(out/'training_base_ids.csv',index=False)
    result = {'analysis_role':'post-review inference-only controls', 'train_base_cases':350,
              'test_base_cases':75,'bootstrap_replicates':5000,'bootstrap_seed':20261009,
              'training_mean_fit':'per-pixel mean over valid training cases only; no condition-specific templates',
              'canonical_maps_sha256':sha(analysis/'paired_test_cohort_maps.npz'),
              'sample_index_sha256':sha(index_path),'script_sha256':sha(__file__),
              'training_target_sources':sources,'mean_field_metrics':records}
    print(table.to_string(index=False), flush=True)
    if args.coordinates:
        sys.path.insert(0,str(root/'scripts'))
        spec=importlib.util.spec_from_file_location('height_eval',root/'scripts/156_run_icbinb_module4_sensor_height_ood.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.torch.cuda.is_available()
        device=module.torch.device('cuda')
        source_inventory=json.loads((analysis/'artifact_inventory.json').read_text())
        artifacts={row['id']:row for row in source_inventory['artifacts']}
        checkpoint=root/artifacts['fno_checkpoint_field']['path']
        stats_path=root/artifacts['fno_training_statistics']['path']
        assert sha(checkpoint)==artifacts['fno_checkpoint_field']['sha256']
        assert sha(stats_path)==artifacts['fno_training_statistics']['sha256']
        stats=module.FNO_EXPORT.load_stats(stats_path)
        heightcohort, rows=module.build_height_rows()
        assert heightcohort.base_sample_id.tolist()==cohort.base_sample_id.tolist()
        nominal_coords=rows[50].coordinate_channels_file.to_numpy().copy()
        predmaps, control_records = {}, []
        for height in [50,35,65]:
            row=rows[height].copy()
            row['coordinate_channels_file']=nominal_coords
            truth, predictions, replay=module.run_fno(row,checkpoint,stats,device,6)
            assert np.allclose(truth['lat'],maps['true_lat'],rtol=0,atol=1e-5)
            for field in ['lat','apd90']:
                pred=predictions[field].astype(np.float64)
                predmaps[f'fno_frozen_coordinate_{height}_{field}']=pred
                if height==50:
                    diff=float(np.max(np.abs(pred-maps['fno_field_pred_'+field])))
                    result['nominal_replay_max_abs_'+field]=diff
                stats= sufficient(maps['true_'+field].astype(np.float64),pred,valid)
                if height==50:
                    metric_diff=abs(float(r2(stats))-float(r2(per_case['fno_'+field])))
                    assert metric_diff<=1e-5, (field,metric_diff)
                    result['nominal_replay_metric_delta_'+field]=metric_diff
                control_records.append({'height_mm':height,'coordinate_height_mm':50,'field':field,
                                       'r2':float(r2(stats))})
            stats=module.FNO_EXPORT.load_stats(stats_path)
            print('Coordinate control completed:',height,flush=True)
        pd.DataFrame(control_records).to_csv(out/'coordinate_freeze_metrics.csv',index=False)
        np.savez_compressed(out/'coordinate_freeze_maps.npz',**predmaps)
        result['coordinate_control']=control_records
        result['checkpoint_sha256']=sha(checkpoint)
        print(pd.DataFrame(control_records).to_string(index=False),flush=True)
    (out/'controls_summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
