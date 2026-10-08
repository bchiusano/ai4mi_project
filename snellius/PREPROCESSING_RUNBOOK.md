# Final SegTHOR preprocessing on Snellius

This job builds the selected deterministic dataset only. It does not create a
CLAHE variant and does not run online augmentation experiments.

## Input

Upload the official clean archive to:

```text
/scratch-shared/$USER/ai4mi_project/source/segthor_train_full.zip
```

The job checks the archive against `data/segthor_train_full.sha256` before
extracting it.

## Run

```bash
cd ~/projects/ai4mi_project
mkdir -p logs
JOB=$(sbatch --parsable snellius/preprocess_full_segthor.sbatch)
echo "$JOB"
squeue -j "$JOB"
```

After the job leaves the queue:

```bash
sacct -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed
cat "logs/segthor_preprocess-${JOB}.out"
cat "logs/segthor_preprocess-${JOB}.err"
```

The final output is:

```text
/scratch-shared/$USER/ai4mi_project/data/SEGTHOR_FULL
```

It contains `img/`, `gt/`, `split.json`, `spacing.pkl`, and
`preprocessing.json`. The repository receives a `data/SEGTHOR_FULL` symlink to
that directory. The job refuses to overwrite an existing dataset and validates
all existing output before reporting success.
