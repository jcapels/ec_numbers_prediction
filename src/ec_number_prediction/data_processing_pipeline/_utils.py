from tqdm import tqdm
import contextlib
import joblib
import re

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def parse_cd_hit(path_to_clstr):
    '''
    Gather the clusters of CD-hit output `path_to_clust` into a dict.
    '''
    # setup regular expressions for parsing
    pat_id = re.compile(r">(.+?)\.\.\.")
    is_center = re.compile(r">(.+?)\.\.\. \*")

    with open(path_to_clstr) as f:
        clusters = {}
        cluster = []
        id_clust = None
        next(f)  # advance first cluster header
        for line in f:
            if line.startswith(">"):
                # if cluster ended, flush seq ids to it
                clusters[id_clust] = cluster
                cluster = []
                continue
            match = pat_id.search(line)
            if match:
                if is_center.search(line):
                    id_clust = match[1]
                else:
                    cluster.append(match[1])
        clusters[id_clust] = cluster
    return clusters


def scale_up_cd_hit(paths_to_clstr):
    '''
    Hierarchically expand CD-hit clusters.

    Parameters
    ----------
    paths_to_clstr: list[str]
        paths to rest of the cd-hit output files, sorted by
        decreasing similarity (first is 100).

    Output
    ------
    clust_now: dict
        id: ids

    '''
    clust_above = parse_cd_hit(paths_to_clstr[0])

    for path in paths_to_clstr[1:]:
        clust_now = parse_cd_hit(path)
        for center in clust_now:
            clust_now[center] += [
                seq
                for a_center in clust_now[center] + [center]
                for seq in clust_above[a_center]
            ]
        clust_above = clust_now

    return clust_above