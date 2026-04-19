from ctypes import *
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from enum import IntEnum, Flag

_libpath = Path(__file__).parent / 'libopenmpt.so'

cdll.LoadLibrary(str(_libpath))
_mpt = CDLL(_libpath)

def _py_logfunc(message: bytes, _) -> None:
    print(message.decode())

def _py_errfunc(error: int, _) -> c_int:
    return c_int(0b10)

class _opaque_module(Structure):
    pass

_logfunc = CFUNCTYPE(c_char_p, c_void_p)(_py_logfunc)
_errfunc = CFUNCTYPE(c_int, c_void_p)(_py_errfunc)
_errno = c_int()
_errmsg = create_string_buffer(256)
_mpt.openmpt_module_create_from_memory2.restype = POINTER(_opaque_module)

MIX_RATE = 48000

class SampleFlags(Flag):
	CHN_16BIT           = 0x01      # 16-bit sample
	CHN_LOOP            = 0x02      # Looped sample
	CHN_PINGPONGLOOP    = 0x04      # Bidi-looped sample
	CHN_SUSTAINLOOP     = 0x08      # Sample with sustain loop
	CHN_PINGPONGSUSTAIN = 0x10      # Sample with bidi sustain loop
	CHN_PANNING         = 0x20      # Sample with forced panning
	CHN_STEREO          = 0x40      # Stereo sample
	CHN_REVERSE         = 0x80      # Start sample playback from sample / loop end (Velvet Studio feature)
	CHN_SURROUND        = 0x100     # Use surround channel
	CHN_ADLIB           = 0x200     # Adlib / OPL instrument is active on this channel


class Command(IntEnum):
	NONE                = 0
	ARPEGGIO            = 1
	PORTAMENTOUP        = 2
	PORTAMENTODOWN      = 3
	TONEPORTAMENTO      = 4
	VIBRATO             = 5
	TONEPORTAVOL        = 6
	VIBRATOVOL          = 7
	TREMOLO             = 8
	PANNING8            = 9
	OFFSET              = 10
	VOLUMESLIDE         = 11
	POSITIONJUMP        = 12
	VOLUME              = 13
	PATTERNBREAK        = 14
	RETRIG              = 15
	SPEED               = 16
	TEMPO               = 17
	TREMOR              = 18
	MODCMDEX            = 19
	S3MCMDEX            = 20
	CHANNELVOLUME       = 21
	CHANNELVOLSLIDE     = 22
	GLOBALVOLUME        = 23
	GLOBALVOLSLIDE      = 24
	KEYOFF              = 25
	FINEVIBRATO         = 26
	PANBRELLO           = 27
	XFINEPORTAUPDOWN    = 28
	PANNINGSLIDE        = 29
	SETENVPOSITION      = 30
	MIDI                = 31
	SMOOTHMIDI          = 32
	DELAYCUT            = 33
	XPARAM              = 34
	FINETUNE            = 35
	FINETUNE_SMOOTH     = 36
	DUMMY               = 37
	NOTESLIDEUP         = 38 # IMF Gxy / PTM Jxy (Slide y notes up every x ticks)
	NOTESLIDEDOWN       = 39 # IMF Hxy / PTM Kxy (Slide y notes down every x ticks)
	NOTESLIDEUPRETRIG   = 40 # PTM Lxy (Slide y notes up every x ticks + retrigger note)
	NOTESLIDEDOWNRETRIG = 41 # PTM Mxy (Slide y notes down every x ticks + retrigger note)
	REVERSEOFFSET       = 42 # PTM Nxx Revert sample + offset
	DBMECHO             = 43 # DBM enable/disable echo
	OFFSETPERCENTAGE    = 44 # PLM Percentage Offset
	DIGIREVERSESAMPLE   = 45 # DIGI reverse sample
	VOLUME8             = 46 # 8-bit volume
	HMN_MEGA_ARP        = 47 # His Master's Noise "mega-arp"
	MED_SYNTH_JUMP      = 48 # MED synth jump / MIDI panning
	AUTO_VOLUMESLIDE    = 49
	AUTO_PORTAUP        = 50
	AUTO_PORTADOWN      = 51
	AUTO_PORTAUP_FINE   = 52
	AUTO_PORTADOWN_FINE = 53
	AUTO_PORTAMENTO_FC  = 54 # Future Composer
	TONEPORTA_DURATION  = 55 # Parameter = how many rows the slide should last
	VOLUMEDOWN_DURATION = 56 # Parameter = how many rows the slide should last
	VOLUMEDOWN_ETX      = 57 # EasyTrax fade-out (parameter = speed, independent of song tempo)


class CMetadata(Structure):
    _fields_ = [('tempo', c_uint32),
                ('speed', c_uint32),
                ('restart_order', c_uint32),
                ('restart_row', c_uint32)]


class CChannelData(Structure):
    _fields_ = [('sample_id', c_uint32),
                ('sample_pos', c_uint32),
                ('sample_inc', c_double),
                ('sample_vol', c_uint32),
                ('cmd', c_uint32),
                ('param', c_uint32)]


class CSample(Structure):
    _fields_ = [('name', c_char_p),
                ('flags', c_uint32),
                ('rate', c_uint32),
                ('loop_start', c_uint32),
                ('loop_end', c_uint32),
                ('sustain_start', c_uint32),
                ('sustain_end', c_uint32),
                ('data_length', c_uint32),
                ('data', c_void_p)]


class Metadata:
    tempo: int
    speed: int
    restart_order: int
    restart_row: int

    def __init__(self, cmeta: CMetadata) -> None:
        self.tempo = cmeta.tempo
        self.speed = cmeta.speed
        self.restart_order = cmeta.restart_order
        self.restart_row = cmeta.restart_row


class ChannelData:
    index: int
    sample_id: int | None
    sample_pos: int
    sample_inc: float
    sample_vol: int
    command: int
    param: int

    def __init__(self, ccd: CChannelData, index: int) -> None:
        self.index = index
        self.sample_id = ccd.sample_id - 1 if ccd.sample_id != 0 else None
        self.sample_pos = ccd.sample_pos
        self.sample_inc = ccd.sample_inc
        self.sample_vol = ccd.sample_vol
        self.command = ccd.cmd
        self.param = ccd.param


@dataclass
class Frame:
    pattern: int
    order: int
    row: int
    tick: int
    samples_per_tick: int
    ticks_per_row: int
    rows_per_beat: int
    rows_per_measure: int
    channels: list[ChannelData]


class Sample:
    name: str
    flags: SampleFlags
    rate: int
    length: int
    loop_start: int
    loop_end: int
    sustain_start: int
    sustain_end: int
    data: np.typing.NDArray

    def __init__(self, csmp: CSample) -> None:
        self.name = csmp.name.decode()
        self.flags = SampleFlags(csmp.flags)
        self.rate = csmp.rate
        self.loop_start = csmp.loop_start
        self.loop_end = csmp.loop_end
        self.sustain_start = csmp.sustain_start
        self.sustain_end = csmp.sustain_end
        # Get data as np array
        dtype = np.int16 if SampleFlags.CHN_16BIT in self.flags else np.int8
        self.data = np.frombuffer(string_at(csmp.data, csmp.data_length), dtype)
        self.length = len(self.data) // 2 if SampleFlags.CHN_STEREO in self.flags else len(self.data)


_mpt.openmpt_module_get_current_channel_lsp.restype = CChannelData
_mpt.openmpt_module_get_meta_lsp.restype = CMetadata
_mpt.openmpt_module_get_sample_lsp.restype = CSample


class MPTMod:
    metadata: Metadata
    frames: list[Frame]
    samples: list[Sample]

    def __init__(self, path: Path | str):
        with open(path, 'rb') as f:
            modfile = f.read()
        # Probe to ensure it's a valid module file
        ret = _mpt.openmpt_probe_file_header(c_uint64(1), c_char_p(modfile), _mpt.openmpt_probe_file_header_get_recommended_size(), c_uint64(len(modfile)), _logfunc, c_void_p(None), _errfunc, c_void_p(None), pointer(_errno), pointer(_errmsg))
        if (ret != 1):
            raise ValueError(f"Module file is invalid")
        # Create module
        mod = _mpt.openmpt_module_create_from_memory2(c_char_p(modfile), c_size_t(len(modfile)), _logfunc, c_void_p(None), _errfunc, c_void_p(None), pointer(_errno), pointer(_errmsg), c_void_p(None))
        # Get metadata
        self.metadata = Metadata(_mpt.openmpt_module_get_meta_lsp(mod))
        # Get frames
        num_channels = _mpt.openmpt_module_get_num_channels(mod)
        frames: list[Frame] = []
        last_row = -1
        tick = 0
        while _mpt.openmpt_module_play_tick(mod):
            pattern = _mpt.openmpt_module_get_current_pattern(mod)
            order = _mpt.openmpt_module_get_current_order(mod)
            row = _mpt.openmpt_module_get_current_row(mod)
            tick = tick + 1 if row == last_row else 0
            samples_per_tick = _mpt.openmpt_module_get_current_samples_per_tick(mod)
            ticks_per_row = _mpt.openmpt_module_get_current_speed(mod)
            rows_per_beat = _mpt.openmpt_module_get_pattern_rows_per_beat(mod, pattern)
            rows_per_measure = _mpt.openmpt_module_get_pattern_rows_per_measure(mod, pattern)
            channels = list(ChannelData(_mpt.openmpt_module_get_current_channel_lsp(mod, i), i) for i in range(num_channels))
            frames.append(Frame(
                pattern,
                order,
                row,
                tick,
                samples_per_tick,
                ticks_per_row,
                rows_per_beat,
                rows_per_measure,
                channels))
            last_row = row
        self.frames = frames
        # Get samples
        num_samples = _mpt.openmpt_module_get_num_samples(mod)
        self.samples = list(Sample(_mpt.openmpt_module_get_sample_lsp(mod, i+1)) for i in range(num_samples))
