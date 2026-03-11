import warnings
from traceback import print_exc
from typing import Any, Sequence, OrderedDict
import networkx as nx
import numpy as np
import pandas as pd
from selfx.backend.features import get_analysis_intervals
from selfx.backend.results import is_stored, get_result


def run_tasks(tasks: Sequence[str], celery_app: Any, interv: Sequence[Any]) -> None:
    """
    Dispatch a chain of celery tasks for a given interval.

    Parameters
    ----------
    tasks:
        List of celery task names present in celery_app.tasks.
    celery_app:
        Celery application instance.
    interv:
        A pair (start, end) of datetime-like objects. They are converted via
        .isoformat() and passed into each task signature.

    Returns
    -------
    None

    Notes
    -----
    - celery is imported lazily because celery import can be heavy.
    - This function only enqueues the chain (apply_async()).
    """
    if not tasks:
        print("No tasks to run")
        return

    from celery import chain  # lazy import

    sigs = [
        celery_app.tasks[tsk].si(interv[0].isoformat(), interv[1].isoformat())
        for tsk in tasks
    ]
    workflow = chain(*sigs)
    print("Sending tasks to celery: ", list(tasks), flush=True)
    workflow.apply_async()


def exist_requested_features(self, feature, system, start, finish):
    features_to_get = self._feature_obj[system][feature].required_features
    if features_to_get is None:
        features_to_get = [feature]
    else:
        features_to_get.append(feature)

    try:
        intervals = get_analysis_intervals(start, finish)
        results = pd.DataFrame(np.nan, index=intervals.keys(), columns=features_to_get)

        for k, interv in intervals.items():
            for f in features_to_get:
                success_f = is_stored(k, f)
                results.loc[k, f] = float(success_f)

        return results
    except Exception as ex:
        print_exc()
        return None

def get_requested_features(self, feature, system, start, finish):
    f_objs = self._feature_obj[system]
    print('Getting required features...')

    required_features = []
    if f_objs[feature].required_features:
        required_features += f_objs[feature].required_features
    features_to_get = [feature] + required_features

    try:
        intervals = get_analysis_intervals(start, finish)
        results = OrderedDict()
        results_failed = OrderedDict()

        for k, interv in intervals.items():
            for f in features_to_get:
                res = get_result(f'{k}/{f}.joblib')
                if res is None or ('status' in res and res['status'] == 'failed'):
                    if k not in results_failed:
                        results_failed[k] = {}
                    results_failed[k][f] = res
                else:
                    if k not in results:
                        results[k] = {}
                    results[k][f] = res
        print('Finished getting required features...')
        return results, results_failed
    except Exception as ex:
        print_exc()
        return None, None


def get_sorted_features(periodic_features):
    graph = nx.DiGraph()
    for obj in periodic_features:
        fname = obj.feature_name()
        if obj.required_features:
            for rf in obj.required_features:
                graph.add_edge(rf, fname)
        else:
            graph.add_node(fname)
    sorted_features = list(nx.topological_sort(graph))
    return sorted_features

def get_required_features(feature, all_features):
    graph = nx.DiGraph()

    feature_name, feature_obj = feature

    parent_nodes = [feature_obj]
    while parent_nodes:
        feature = parent_nodes.pop()
        if feature.required_features:
            for rf in feature.required_features:
                if rf not in all_features:
                    raise Exception(f'Feature {rf} not in available features.')
                graph.add_edge(rf, feature.feature_name())
                parent_nodes.append(all_features[rf])
        else:
            graph.add_node(feature.feature_name())
    return list(nx.topological_sort(graph))

def perform_requested_features(feature_objects, celery_app, feature, system, start, finish):
    """
    Perform requested features.
    :param system:
    :param data:
    :param start:
    :param finish:
    :return:
    """

    if start is None or finish is None:
        warnings.warn("Start or Finish is None when calling perform requested features.")
        return

    f_objs = feature_objects
    print('Performing requested features: ')
    features_to_perform = get_required_features((feature, f_objs[feature]), f_objs)
    print('Features to perform: ', features_to_perform)
    try:
        intervals = get_analysis_intervals(start, finish)
        print(f'In the interval [{start}, {finish}) the following subintervals will be analyzed: ')
        print(intervals)
        for k, interv in intervals.items():
            features_to_perform_tasks = []
            for f in features_to_perform:
                f = system + f
                if is_stored(k, f):
                    print(f"Skipping feature performing, it exists: {f} in {k}")
                    continue
                else:
                    print(f"Feature performing, it does not exist: {f} in {k}")
                features_to_perform_tasks.append(f)

            run_tasks(features_to_perform_tasks, celery_app, interv)
    except Exception as ex:
        print_exc()