python -m backend.rag.vectorstore                                                     23:01:29 
Building vector store with 3405 documents at 'D:\Github\langchain_practice\backend\data\chroma_db'...
Using HuggingFace
Traceback (most recent call last):
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\utils\import_utils.py", line 2317, in __getattr__
    module = self._get_module(self._class_to_module[name])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\utils\import_utils.py", line 2347, in_get_module
    raise e
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\utils\import_utils.py", line 2345, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in_gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in_find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\modeling_utils.py", line 70, in <module>
    from .loss.loss_utils import LOSS_MAPPING
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\loss\loss_utils.py", line 21, in <module>
    from .loss_d_fine import DFineForObjectDetectionLoss
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\loss\loss_d_fine.py", line 21, in <module>
    from .loss_for_object_detection import (
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\loss\loss_for_object_detection.py", line 32, in <module>
    from transformers.image_transforms import center_to_corners_format
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\image_transforms.py", line 22, in <module>
    from .image_utils import (
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\image_utils.py", line 55, in <module>
    from torchvision.transforms import InterpolationMode
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\torchvision\__init__.py", line 10, in <module>
    from torchvision import_meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\torchvision\_meta_registrations.py", line 163, in <module>
    @torch.library.register_fake("torchvision::nms")
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\torch\library.py", line 1063, in register
    use_lib._register_fake(
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\torch\library.py", line 211, in_register_fake
    handle = entry.fake_impl.register(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\torch\_library\fake_impl.py", line 50, in register
    if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: operator torchvision::nms does not exist

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\langchain_huggingface\embeddings\huggingface.py", line 68, in __init__
    import sentence_transformers  # type: ignore[import]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\sentence_transformers\__init__.py", line 15, in <module>
    from sentence_transformers.cross_encoder import (
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\sentence_transformers\cross_encoder\__init__.py", line 3, in <module>
    from .CrossEncoder import CrossEncoder
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\sentence_transformers\cross_encoder\CrossEncoder.py", line 16, in <module>
    from transformers import (
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\transformers\utils\import_utils.py", line 2320, in __getattr__
    raise ModuleNotFoundError(
ModuleNotFoundError: Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\Github\langchain_practice\backend\rag\vectorstore.py", line 68, in <module>
    build_vectorstore(docs, persist_directory=target_dir)
  File "D:\Github\langchain_practice\backend\rag\vectorstore.py", line 33, in build_vectorstore
    embeddings = get_embeddings()
                 ^^^^^^^^^^^^^^^^
  File "D:\Github\langchain_practice\backend\rag\embeddings.py", line 35, in get_embeddings
    return HuggingFaceEmbeddings(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\miniconda3\envs\langchain_env\Lib\site-packages\langchain_huggingface\embeddings\huggingface.py", line 74, in __init__
    raise ImportError(msg) from exc
ImportError: Could not import sentence_transformers python package. Please install it with `pip install sentence-transformers`
