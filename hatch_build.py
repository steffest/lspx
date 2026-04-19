import os
import importlib.util
import subprocess
from typing import Any
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data['pure_python'] = False
        build_data['infer_tag'] = True
        if not os.environ.get('CIBUILDWHEEL', None):
            env = os.environ | {
                'SHARED_LIB': '1',
                'STATIC_LIB': '0',
                'EXAMPLES': '0',
                'OPENMPT123': '0',
                'TEST': '0',
                'MODERN': '1',
                'NO_ZLIB': '1',
                'NO_MPG123': '1',
                'NO_OGG': '1',
                'NO_VORBIS': '1',
                'NO_VORBISFILE': '1',
                'NO_MINIZ': '1',
                'NO_MINIMP3': '1',
                'NO_STBVORBIS': '1',
                'NO_PORTAUDIO': '1',
                'NO_PORTAUDIOCPP': '1',
                'NO_PULSEAUDIO': '1',
                'NO_SDL2': '1',
                'NO_FLAC': '1',
                'NO_SNDFILE': '1',
                'SHARED_SONAME': '0'
            }
            if os.name == 'nt':
                env['CPPFLAGS'] = '-DLIBOPENMPT_BUILD_DLL'
            cpus = os.cpu_count()
            if cpus is None:
                cpus = 2
            subprocess.run(['make', '-j', str(cpus)], cwd='openmpt-lspx', check=True, env=env)


class VersionMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        path = Path(__file__).parent / 'src' / 'lspx' / 'version.py'
        spec = importlib.util.spec_from_file_location('version', path)
        assert spec and spec.loader
        version = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version)
        metadata['version'] = version.VERSION
