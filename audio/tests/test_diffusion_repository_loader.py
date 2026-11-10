import multiprocessing
import pickle
from pathlib import Path

from audio.diffusion.loading import load_repository_classes
from audio.tests.forkserver_repository_probe import unpickle_processor_class


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_loader_uses_modified_worktree_not_checkpoint_remote_code():
    classes = load_repository_classes(
        REPO_ROOT, package_name="_mawjais_diffusion_repository_loader_test"
    )
    assert classes.model_class.__module__.endswith(".modeling_xllm_aut")
    assert classes.processor_class.__module__.endswith(".processing_xllm_aut")
    assert classes.config_class.__module__.endswith(".configuration_xllm_aut")

    model_source = Path(__import__(classes.model_class.__module__, fromlist=["x"]).__file__)
    processor_source = Path(
        __import__(classes.processor_class.__module__, fromlist=["x"]).__file__
    )
    assert model_source.resolve() == (REPO_ROOT / "audio/model/modeling_xllm_aut.py").resolve()
    assert processor_source.resolve() == (
        REPO_ROOT / "audio/model/processing_xllm_aut.py"
    ).resolve()


def test_default_repository_package_is_importable_by_forkserver_worker():
    classes = load_repository_classes(REPO_ROOT)
    assert classes.package_name == "_mawjais_diffusion_repository"

    # Serialize the same class reference held inside the real processor object
    # captured by MawJaisDiffusionCollator. The worker receives bytes so the
    # decisive import happens only during pickle.loads in its fresh interpreter.
    serialized_class = pickle.dumps(classes.processor_class)
    context = multiprocessing.get_context("forkserver")
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=unpickle_processor_class,
        args=(serialized_class, sending_connection),
    )
    process.start()
    sending_connection.close()
    try:
        assert receiving_connection.poll(60), "forkserver worker did not answer"
        result = receiving_connection.recv()
        process.join(timeout=60)
        assert process.exitcode == 0
    finally:
        receiving_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert result.get("ok") is True, result.get("traceback")
    assert result["module"] == (
        "_mawjais_diffusion_repository.processing_xllm_aut"
    )
    assert result["name"] == "XllmAuTProcessor"
    assert Path(result["source"]).resolve() == (
        REPO_ROOT / "audio/model/processing_xllm_aut.py"
    ).resolve()
