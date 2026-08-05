from __future__ import annotations
import struct
import asyncio
import collections
from typing import Any
from enum import IntEnum
from pathlib import Path
from hashlib import md5
import math
import numpy as np
from dataclasses import dataclass, field
from . import mpt

def _vol_to_db(vol: int) -> float:
    """Takes a MPT volume value and returns the equivalent volume in
    decibels."""
    if vol == 0:
        return -math.inf
    return math.log10(vol/16384) * 20

def _db_to_vol(gain: float) -> int:
    """Takes a decibel value and returns the nearest corresponding Paula volume
    value."""
    if gain == -math.inf:
        return 0
    return round(10 ** (gain / 20) * 64)

def _magnitude_to_db(mag: float) -> float:
    """Takes a volume magnitude (e.g. multiplier to adjust volume by) and
    returns the dB amount to adjust gain in order to achieve that ratio."""
    return math.log10(mag) * 20


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    """A config specification for the loaded module. This holds options
    specified in command line arguments (highest priority), a config file, or
    default values."""

    ntsc: bool = False
    """Whether to use NTSC clock rate (instead of PAL) to calculate Amiga
    period values."""

    min_rate: int = 28604
    """The target minimum sample rate at which samples will play. LSPX will
    attempt to downsample each sample such that the lowest pitch it ever
    reaches in your module is equal to this sample rate. Note that depending on
    the pitch range of the sample, it may be downsampled further to ensure it
    doesn't overshoot ``max_rate``."""

    max_rate: int = 28604
    """The maximum sample rate at which samples will ever play. LSPX will
    downsample each sample such that the highest pitch it ever reaches is at or
    below this sample rate."""

    mixer_hw_channel: int = 0
    """For mixer processing. The Amiga hardware channel being used for the
    mixer. Any sample played on this channel is subject to additional
    restrictions to ensure mixer compatibility, e.g. it must always play at the
    same pitch and volume on this channel."""

    mixer_sw_channels: int = 1
    """For mixer processing. The number of software channels being mixed by the
    mixer. When this is 1, mixer processing is disabled. Otherwise, samples
    will have their peak level adjusted to be possible to mix into this number
    of software channels. Samples that actually play on the mixer hardware
    channel will have their waveform adjusted, while other samples will retain
    their original peaks and instead have volume commands adjusted to conform
    to the overall lower volume, for maximum quality retention."""

    mixer_rate: int = 11025
    """For mixer processing. The sample rate at which the mixer channel plays.
    Any sample appearing on the mixer channel will be downsampled to reach this
    exact sample rate whenever it's played on the mixer channel."""

    mixer_experimental: bool = False
    sox_path: Path | None = None

    getpos: bool = False
    """Whether to emit LSP GetPos commands at module sequence (order)
    boundaries."""

    beat_events: bool = False
    """Whether to emit explicit beat markers using the GetPos escape-command
    extension understood by the bundled player."""

    @property
    def clock(self) -> int:
        """Returns the clock rate used to calculate period, based on PAL or
        NTSC setting."""
        return 3579545 if self.ntsc else 3546895

    @property
    def cia_clock(self) -> float:
        """Returns the CIA clock value used to calculate number of CIA timer
        ticks per frame, based on PAL or NTSC setting."""
        # From the comments in HRM example 8520_timing.asm
        return 1.3968255742 if self.ntsc else 1.4096836811

    @property
    def mixer(self) -> bool:
        """Returns whether or not mixer processing is enabled."""
        return self.mixer_sw_channels > 1

    @property
    def mixer_gain_adjust(self) -> float:
        """Returns how much to adjust gain on each sample based on current
        mixer settings."""
        return _magnitude_to_db(1 / self.mixer_sw_channels) if self.mixer else 0

    def __post_init__(self) -> None:
        # Validation
        if self.mixer_sw_channels < 1 or self.mixer_sw_channels > 4:
            raise ValueError(f"Mixer SW channels is {self.mixer_sw_channels} but must be between 1 and 4")
        if self.min_rate <= 0:
            raise ValueError(f"Min rate is {self.min_rate} but must be greater than 0")


@dataclass(frozen=True, slots=True)
class SampleConfig:
    """A config specification for a given sample. These options are specified
    in the module's config file."""

    # TODO: Loop correction option, slide loop points forward/back a few
    # samples to find either closest sample to 0 or closest to each other
    # TODO: Option to ramp down note cuts to avoid pop
    mod: ModuleConfig
    """A reference to the ``ModuleConfig`` for the module this sample is in."""

    cutoff: float = 0.0 # TODO
    gain: float = 0.0 # TODO
    quality: float = 1.0
    """Unconditional quality multiplier. After all other downsample
    calculations, the sample rate will be further reduced by this factor."""

    @staticmethod
    def from_dict(mod: ModuleConfig, data: dict) -> SampleConfig:
        """Returns a ``SampleConfig`` with the values specified in the given
        dict (or default values if absent)."""
        return SampleConfig(mod, **data)

    def __post_init__(self) -> None:
        # Validation
        if self.cutoff < 0 or self.cutoff > 1:
            raise ValueError(f"Cutoff is {self.cutoff} but must be between 0 and 1")
        if self.quality <= 0 or self.quality > 1:
            raise ValueError(f"Quality is {self.quality} but must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Config:
    """A full config specification containing configs for the loaded module and
    its samples."""

    mod: ModuleConfig
    """Module-level config options."""

    default: SampleConfig
    """The default sample config options to be used for all samples that don't
    manually specify the option for themselves."""

    samples: dict[str, SampleConfig]
    """Sample config options for each sample, accessible by sample name."""

    @staticmethod
    def from_dict(data: dict) -> Config:
        """Returns a ``Config`` with the values specified in the given dict (or
        default values if absent)."""
        mod = ModuleConfig(**{k: v for k, v in data.get('module', {}).items()})
        default_dict = data.get('default', {})
        default = SampleConfig.from_dict(mod, default_dict)
        samples_data = data.get('samples', {})
        samples = {k: SampleConfig.from_dict(mod, default_dict | v) for k, v in samples_data.items()}
        return Config(mod, default, samples)

    def get(self, name: str) -> SampleConfig:
        """Gets a ``SampleConfig`` by sample name (or default if absent)."""
        return self.samples.get(name, self.default)


@dataclass(frozen=True, slots=True)
class Instrument:
    """An abstraction of an LSP instrument, which consists of a sample and a
    start offset."""

    sample: Sample
    offset: int


@dataclass(frozen=True, slots=True)
class MInstrument:
    """An Instrument that contains the final MSample and an adjusted start
    offset based on the downsample factor."""

    sample: MSample
    offset: int

    def __hash__(self) -> int:
        return hash((self.sample.id, self.offset))


@dataclass(slots=True)
class Frame:
    """A single frame of music playback, containing the IR for each channel and
    some metadata."""

    ir: list[ChannelIR]
    """The list of IRs for each channel."""

    new_row: bool
    """Whether this frame is the start of a new row."""

    new_beat: bool
    """Whether this frame is the start of a new beat."""

    new_measure: bool
    """Whether this frame is the start of a new measure."""

    event: int | None

    def __init__(self, mod: Module, frame: mpt.Frame) -> None:
        self.ir = [ChannelIR(mod, chan) for chan in frame.channels]
        self.new_row = frame.tick == 0
        self.new_beat = self.new_row and frame.row % frame.rows_per_beat == 0
        self.new_measure = self.new_row and frame.row % frame.rows_per_measure == 0
        self.event = None
        if mod.config.mod.mixer_experimental and self.new_row:
            chan = frame.channels[mod.config.mod.mixer_hw_channel]
            if chan.command == mpt.Command.XPARAM:
                self.event = chan.param


@dataclass(slots=True)
class MFrame:
    """A single frame of music playback after the IR has been processed into
    MIR."""

    mir: list[ChannelMIR]
    """The list of MIRs for each channel."""

    new_row: bool
    """Whether this frame is the start of a new row."""

    new_beat: bool
    """Whether this frame is the start of a new beat."""

    new_measure: bool
    """Whether this frame is the start of a new measure."""

    command: int
    """The final LSP command value for this frame."""

    event: int | None

    def __init__(self, mod: Module, frame: Frame, prev: MFrame | None):
        self.new_row = frame.new_row
        self.new_beat = frame.new_beat
        self.new_measure = frame.new_measure
        self.event = frame.event
        config = mod.config.mod
        mir = []
        for i, chan in enumerate(frame.ir):
            prev_chan = prev.mir[i] if prev is not None else None
            mixer = config.mixer and i == config.mixer_hw_channel
            mir.append(ChannelMIR(mod, chan, prev_chan, mixer))
        self.mir = mir
        # Get the voice code for this frame
        codes: list[MFrame.Code] = []
        for chan in self.mir:
            code = self.Code.NONE
            if chan.inst is None and chan.prev is not None and chan.prev.inst is not None:
                code = self.Code.RESET_LEN
            elif chan.inst is not None:
                code = self.Code.PLAY_INSTRUMENT
            codes.append(code)
        # Mixer channel only needs PLAY_INSTRUMENT code, others are ignored
        if config.mixer:
            i = config.mixer_hw_channel
            if codes[i] != self.Code.PLAY_INSTRUMENT:
                codes[i] = self.Code.NONE
        # Experimental mixer uses PLAY_WITHOUT_NOTE in place of PLAY_INSTRUMENT
        # to free up the extra bit for specifying event on mixer channel
        if config.mixer_experimental:
            for i in range(len(codes)):
                if codes[i] == self.Code.PLAY_INSTRUMENT:
                    codes[i] = self.Code.PLAY_WITHOUT_NOTE
        inst = vol = per = 0
        for code, chan in zip(codes[::-1], self.mir[::-1]):
            inst <<= 2
            vol <<= 1
            per <<= 1
            inst |= code
            vol |= int(chan.vol is not None)
            per |= int(chan.per is not None)
        if config.mixer_experimental:
            vol |= int(self.new_beat) << config.mixer_hw_channel
            per |= int(self.new_measure) << config.mixer_hw_channel
            if self.event is not None:
                inst |= 1 << (config.mixer_hw_channel * 2)
        self.command = inst << 8 | vol << 4 | per


    class Code(IntEnum):
        """ An enum of the possible instrument commands."""
        NONE = 0
        """ No action."""
        RESET_LEN = 1
        """ Reset AUDxPT and AUDxLEN to the instrument's loop values. This
            happens the frame after the instrument was played."""
        PLAY_WITHOUT_NOTE = 2
        """ Set an instrument without actually playing a note."""
        PLAY_INSTRUMENT = 3
        """ Set and play an instrument."""


@dataclass(slots=True)
class ChannelIR:
    """The playback instructions for a single channel on a single frame."""

    inst: Instrument | None
    """The ``Instrument`` being played on this frame."""

    pitch: float | None
    """The pitch being played on this frame. The value is a multiplier relative
    to ``mpt.MIX_RATE``, e.g. a pitch of 0.5 indicates the sample is being
    played at a sample rate of half that value."""

    gain: float | None
    """The gain (volume in dB) being played on this frame."""

    def __init__(self, mod: Module, chan: mpt.ChannelData) -> None:
        inst = None
        # TODO: Detect new instrument/note from other methods like sample cues
        if chan.sample_id is not None and ((chan.command == mpt.Command.OFFSET and chan.sample_pos == chan.param * 256) or chan.sample_pos == 0):
            offset = chan.param * 256 if chan.command == mpt.Command.OFFSET else 0
            inst = Instrument(mod.samples[chan.sample_id], offset)
        self.inst = inst
        self.pitch = chan.sample_inc if chan.sample_id is not None else None
        self.gain = _vol_to_db(chan.sample_vol) if chan.sample_id is not None else None


@dataclass(slots=True)
class ChannelMIR:
    """The playback instructions for a single channel on a single frame. The
    data is lowered to Amiga-specific machine values. Furthermore, the
    instructions are filtered such that redundant period/volume/instrument
    commands are omitted."""

    inst: MInstrument | None
    """The instruction on this frame to play a new instrument."""

    per: int | None
    """The instruction on this frame to set the Amiga period."""

    vol: int | None
    """The instruction on this frame to set the Amiga volume."""

    prev: ChannelMIR | None
    """The previous MIR on this channel."""

    active_inst: MInstrument | None
    """The current instrument actively being played on this channel."""

    active_per: int | None
    """The current period actively being played on this channel."""

    active_vol: int | None
    """The current volume actively being played on this channel."""

    def __init__(self, mod: Module, ir: ChannelIR, prev: ChannelMIR | None, mixer: bool) -> None:
        self.prev = prev
        if prev is not None:
            active_inst = prev.active_inst
            active_per = prev.active_per
            active_vol = prev.active_vol
        else:
            active_inst = None
            active_per = None
            active_vol = None
        inst = None
        per = None
        vol = None
        # Instrument
        if ir.inst is not None:
            msmp = mod.msamples[ir.inst.sample.id]
            offset = int(ir.inst.offset * msmp.downsample_factor)
            inst = MInstrument(msmp, offset)
            active_inst = inst
        self.inst = inst
        self.active_inst = active_inst
        # We never use period or volume commands on mixer channel
        if mixer:
            self.per = None
            self.vol = None
            self.active_per = None
            self.active_vol = None
            return
        # Period
        if active_inst is not None and ir.pitch is not None:
            freq = mpt.MIX_RATE * active_inst.sample.downsample_factor * abs(ir.pitch)
            per = round(mod.config.mod.clock / freq)
            active_per = per
        # Volume
        if ir.gain is not None:
            # If sample isn't used on mixer channel, it needs gain adjusted to
            # bring its volume in line with the mixer channel's reduced peak
            if mod.config.mod.mixer and (active_inst is None or not active_inst.sample.mixer):
                vol = _db_to_vol(ir.gain + mod.config.mod.mixer_gain_adjust)
            else:
                vol = _db_to_vol(ir.gain)
            active_vol = vol
        # Only apply new commands if different from previous command
        if prev is not None:
            if per == prev.active_per: per = None
            if vol == prev.active_vol: vol = None
        self.per = per
        self.vol = vol
        self.active_per = active_per
        self.active_vol = active_vol


@dataclass(frozen=True, slots=True)
class Sample:
    """A module sample, containing the sample itself along with metadata
    related to how the sample is played throughout the module."""

    sample: mpt.Sample
    """The base MPT sample structure."""

    config: SampleConfig
    """The config for this sample that was loaded from the config file."""

    id: int
    """The unique sample ID number."""

    mixer: bool
    """Whether the sample is used on the mixer channel (always False if the
    mixer is not being used)."""

    lowest_pitch: float
    """The lowest pitch at which the sample is ever played in the module
    (pitch is relative to ``mpt.MIX_RATE``)."""

    highest_pitch: float
    """The highest pitch at which the sample is ever played in the module
    (pitch is relative to ``mpt.MIX_RATE``)."""

    lowest_gain: float
    """The lowest gain at which the sample is ever played in the module."""

    highest_gain: float
    """The highest gain at which the sample is ever played in the module."""

    earliest_pos: int
    """The earliest offset this sample ever reaches during module playback."""

    latest_pos: int
    """The latest offset this sample ever reaches during module playback."""

    downsample_factor: float
    """The amount that this sample is to be downsampled during processing,
    based on playback data and user config."""

    @dataclass(slots=True)
    class _State:
        used: bool = False
        pitch_min: float = math.inf
        pitch_max: float = -math.inf
        vol_min: int = 16384
        vol_max: int = 0
        pos_min: int = 2 ** 32 - 1
        pos_max: int = 0
        mx_pitch: None | float | ValueError = None
        mx_vol: None | int | ValueError = None

        def parse_chan(self, chan: mpt.ChannelData, mixer: bool, samples_per_tick: int):
            # TODO: If a sample is at zero volume, it should not have min/max
            # data updated, and should not be considered used until it's at
            # a non-zero volume
            self.used = True
            # abs because reverse and pingpong samples have a negative pitch
            pitch = abs(chan.sample_inc)
            self.pitch_min = min(self.pitch_min, pitch)
            self.pitch_max = max(self.pitch_max, pitch)
            vol = chan.sample_vol
            self.vol_min = min(self.vol_min, vol)
            self.vol_max = max(self.vol_max, vol)
            # Calculate sample pos at the end of this tick as the true max pos
            pos = chan.sample_pos
            end_pos = pos + int(samples_per_tick * pitch)
            self.pos_min = min(self.pos_min, pos)
            self.pos_max = max(self.pos_max, end_pos)
            # Validate pitch/vol never changes on mixer channel
            if mixer:
                if self.mx_pitch is None:
                    self.mx_pitch = pitch
                elif isinstance(self.mx_pitch, float) and pitch != self.mx_pitch:
                    self.mx_pitch = ValueError()
                if self.mx_vol is None:
                    self.mx_vol = vol
                elif isinstance(self.mx_vol, int) and vol != self.mx_vol:
                    self.mx_vol = ValueError()

    @staticmethod
    def _calc_downsample_factor(smp_config: SampleConfig, state: Sample._State) -> float:
        orig_rate = mpt.MIX_RATE
        min_rate = smp_config.mod.min_rate
        max_rate = smp_config.mod.max_rate
        # First, try targeting min_rate. If the highest pitch still overshoots
        # max_rate, we instead need to downsample to target max_rate.
        factor = min_rate / (orig_rate * state.pitch_min)
        if orig_rate * factor * state.pitch_max > max_rate:
            factor = max_rate / (orig_rate * state.pitch_max)
        # We never upsample
        factor = min(factor, 1.0)
        return factor * smp_config.quality

    @staticmethod
    def init_if_used(mod: Module, sample_id: int) -> Sample | None:
        """Returns a new ``Sample`` if the sample is ever played during module
        playback, otherwise ``None``."""
        mpt_mod = mod.mpt_mod
        sample = mpt_mod.samples[sample_id]
        mod_config = mod.config.mod
        smp_config = mod.config.get(sample.name)
        state = Sample._State()
        # Parse frames
        for frame in mpt_mod.frames:
            for i, chan in enumerate(frame.channels):
                if chan.sample_id != sample_id:
                    continue
                mixer = mod_config.mixer and i == mod_config.mixer_hw_channel
                state.parse_chan(chan, mixer, frame.samples_per_tick)
        if not state.used:
            return None
        # TODO: Maybe in these cases, we can optionally split mixer sample into
        # a unique sample to lift these restrictions. Or just disable this
        # validation
        if isinstance(state.mx_pitch, ValueError):
            raise ValueError(f"Sample \"{sample.name}\" played on the mixer channel must always play at the same pitch on that channel")
        if isinstance(state.mx_vol, ValueError):
            raise ValueError(f"Sample \"{sample.name}\" played on the mixer channel must always play at the same volume on that channel")
        if state.mx_vol is not None and state.vol_max > state.mx_vol:
            raise ValueError(f"Sample \"{sample.name}\" must never exceed the volume it's being played at on the mixer channel")
        # Calc downsample factor
        # If used on mixer channel, we must always downsample such that the
        # pitch played on that channel equals the mixer rate.
        if state.mx_pitch is not None:
            factor = mod_config.mixer_rate / (mpt.MIX_RATE * state.mx_pitch)
        else:
            factor = Sample._calc_downsample_factor(smp_config, state)
        return Sample(
                sample,
                smp_config,
                sample_id,
                state.mx_pitch is not None,
                state.pitch_min,
                state.pitch_max,
                _vol_to_db(state.vol_min),
                _vol_to_db(state.vol_max),
                state.pos_min,
                min(state.pos_max, sample.length),
                factor)


@dataclass(frozen=True, slots=True)
class MSample:
    """A sample after its data has been processed into the final format for
    inclusion in the LSP soundbank."""

    id: int
    """The unique sample ID number."""

    mixer: bool
    """Whether the sample is used on the mixer channel (always False if the
    mixer is not being used)."""

    loop_start: int
    """The sample's loop start offset."""

    loop_end: int
    """The sample's loop end offset."""

    downsample_factor: float
    """The amount that this sample has been downsampled from its original."""

    data: bytes = field(repr=False)
    """The raw sample data."""

    @staticmethod
    async def init_async(smp: Sample) -> MSample:
        """Returns a fully-processed ``MSample`` given a source ``Sample``. The
        function is async so that ``sox`` can be invoked on its own thread,
        allowing samples to be processed in parallel."""
        factor = smp.downsample_factor
        rate = round(mpt.MIX_RATE * factor)
        in_data = smp.sample.data[:smp.latest_pos]
        sox_path = 'sox' if smp.config.mod.sox_path is None else str(smp.config.mod.sox_path)
        sox_args = MSample._get_sox_args(smp, rate)
        proc = await asyncio.create_subprocess_exec(sox_path, *sox_args.split(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        proc_out, proc_err = await proc.communicate(bytes(in_data))
        if proc.returncode != 0:
            raise IOError(f'sox command failed: {proc_err.decode()}')
        data = np.frombuffer(proc_out, dtype=np.int8)
        if smp.mixer:
            sw_channels = smp.config.mod.mixer_sw_channels
            data = np.trunc(data / sw_channels).astype(np.int8)
        data = bytes(data)
        # Post-processing
        loop_start = 0
        loop_end = 0
        # Bake in ping-pong loop if needed
        if mpt.SampleFlags.CHN_LOOP in smp.sample.flags:
            loop_start = int(smp.sample.loop_start * factor)
            loop_end = int(smp.sample.loop_end * factor)
            if mpt.SampleFlags.CHN_PINGPONGLOOP in smp.sample.flags:
                data += data[loop_end:loop_start:-1]
                loop_end += loop_end - loop_start
        # Ensure first two bytes of non-looping sample are zero
        else:
            while any(data[:2]):
                data = bytes(1) + data
        # Pad to 2-byte alignment (4-byte if used in mixer)
        pad_amount = (4 - len(data)) % 4 if smp.mixer else len(data) % 2
        data += bytes(pad_amount)
        # If used in mixer, add MFX struct to beginning of sample
        if smp.mixer:
            if loop_end == 0: loop = 1
            elif loop_start == 0: loop = -1
            else: loop = -2
            mfx_data = struct.pack('>LhhL',
                                   len(data),
                                   loop,
                                   7,
                                   loop_start)
            data = mfx_data + data
        return MSample(smp.id, smp.mixer, loop_start, loop_end, factor, data)

    @staticmethod
    def _get_sox_args(smp: Sample, rate: int) -> str:
        # Global args: Repeatable PRNG mode (-R), disable dither (-D)
        global_args = '-R -D'
        # TODO: Ensure consistency of input sample format between mod/xm/mptm
        bitrate = 16 if mpt.SampleFlags.CHN_16BIT in smp.sample.flags else 8
        channels = 2 if mpt.SampleFlags.CHN_STEREO in smp.sample.flags else 1
        in_args = f'-t raw -r {mpt.MIX_RATE} -b {bitrate} -c {channels} -e signed-integer'
        out_args = '-t raw -b 8 -c 1 -e signed-integer'
        effect_args = f'rate -v {rate}'
        args = f'{global_args} {in_args} - {out_args} - {effect_args}'
        return args


@dataclass(frozen=True, slots=True)
class LspInstrument:
    """An instrument as it appears in the final LSP instrument table."""

    sample_offset: int
    """The offset of the sample to play from the start of the LSP soundbank."""

    word_count: int
    """The length of the instrument in words (16-bit values)."""

    loop_offset: int
    """The sample loop offset, relative to the start of the LSP soundbank."""

    loop_word_count: int
    """The length of the loop in words (16-bit values)."""

    def to_bytes(self) -> bytes:
        """Returns the final packed struct of the instrument."""
        return struct.pack('>LHLH',
                           self.sample_offset,
                           self.word_count,
                           self.loop_offset,
                           self.loop_word_count)


class Module:
    """The tracker module with all config, frames, and samples processed for
    inclusion in the final LSP output."""

    config: Config
    """Config options provided by the user via command-line arguments or a
    config file."""

    mpt_mod: mpt.MPTMod
    """The module data as it was received from ``libopenmpt``."""

    samples: dict[int, Sample]
    """The original samples used in this module, indexed by unique ID."""

    msamples: dict[int, MSample]
    """The fully-processed versions of the samples used in this module, indexed
    by unique ID."""

    frames: list[Frame]
    """A list of playback frames (ticks) played by this module, from start to
    finish, containing state information on each channel on each frame."""

    mframes: list[MFrame]
    """The module's playback frames (ticks) after they have been lowered to
    Amiga-specific data."""

    def __init__(self, input_file: Path | str, config: dict[str, Any] = {}) -> None:
        self.config = Config.from_dict(config)
        mod = mpt.MPTMod(input_file)
        self.mpt_mod = mod
        # Process samples
        self.samples = {}
        for id in range(len(mod.samples)):
            if smp := Sample.init_if_used(self, id):
                self.samples[id] = smp
        # Process IR
        self.frames = [Frame(self, frame) for frame in mod.frames]
        # Process Msamples
        msamples = asyncio.run(self._get_msamples(self.samples))
        # Sort dict (for consistency between runs on the same file)
        self.msamples = dict(sorted(msamples.items()))
        # Process MIR
        self.mframes = []
        for f in range(len(self.frames)):
            prev = self.mframes[f-1] if f > 0 else None
            self.mframes.append(MFrame(self, self.frames[f], prev))

    @staticmethod
    def _get_loop_index(mod: mpt.MPTMod) -> int:
        for i, frame in enumerate(mod.frames):
            if frame.order == mod.metadata.restart_order and frame.row == mod.metadata.restart_row:
                return i
        raise ValueError("Cannot find restart position")

    @staticmethod
    async def _get_msamples(samples: dict[int, Sample]) -> dict[int, MSample]:
        tasks: dict[int, asyncio.Task[MSample]] = {}
        async with asyncio.TaskGroup() as tg:
            for id, smp in samples.items():
                tasks[id] = tg.create_task(MSample.init_async(smp))
        return {id: task.result() for id, task in tasks.items()}

    def build(self) -> tuple[bytes, bytes]:
        """Builds this object into the final LSP files. The first item in the
        returned tuple is the soundbank (.lsbank), and the second is the score
        (.lsmusic)."""
        commands = [f.command for f in self.mframes]
        # Generate command table out of all unique LSP frame commands, sorted
        # by most common (because commands >= index 256 take longer to execute)
        command_counts = collections.Counter(commands)
        cmd_table = [c[0] for c in command_counts.most_common()]
        # Now, next three unused commands become magic numbers for special
        # commands. These are ensured >= index 256 so LSP can save cycles by
        # only checking for magic numbers if index >= 256.
        if len(cmd_table) < 255:
            cmd_table += [0] * (255 - len(cmd_table))
        i = 1
        while i in cmd_table: i += 1
        cmd_table.append(esc_rewind := i)
        while i in cmd_table: i += 1
        cmd_table.append(esc_setbpm := i)
        while i in cmd_table: i += 1
        cmd_table.append(esc_getpos := i)
        # Insert empty command at multiples of 256. LSP command index is 8-bit,
        # so 00 means "increment 256" and not "index 0".
        for i in range(0, len(cmd_table), 256):
            cmd_table.insert(i, 0)
        # Setup final data stream by iterating through frames
        minsts = list(set(chan.inst for frame in self.mframes for chan in frame.mir
                          if chan.inst is not None))
        loop_index = self._get_loop_index(self.mpt_mod)
        byte_stream: list[int] = []
        word_stream: list[int] = []
        byte_loop = 0
        word_loop = 0
        beat_index = 0
        bpm = self.mpt_mod.frames[0].tempo
        if not 0 < bpm <= 0xffff:
            raise ValueError(f"Initial BPM {bpm} does not fit in the LSP header")
        previous_bpm = bpm
        for i, frame in enumerate(self.mframes):
            if i == loop_index:
                byte_loop = len(byte_stream)
                word_loop = len(word_stream) * 2
            # GetPos commands update the replay's current sequence number
            # before the first audible tick of each module order.
            if self.config.mod.getpos:
                mpt_frame = self.mpt_mod.frames[i]
                previous_order = self.mpt_mod.frames[i - 1].order if i > 0 else None
                if mpt_frame.order != previous_order:
                    if not 0 <= mpt_frame.order < 0x80:
                        raise ValueError(
                            f"GetPos sequence position {mpt_frame.order} exceeds the supported range 0..127"
                        )
                    cmd_index = cmd_table.index(esc_getpos)
                    h, l = cmd_index // 256, cmd_index % 256
                    byte_stream += [0] * h + [l, mpt_frame.order]
            # Values with bit 7 set transport explicit beat markers without
            # changing the LSP score header layout. The bundled player routes
            # them separately from GetPos sequence values. The low seven bits
            # contain the song's beat number, wrapping every 128 beats.
            if self.config.mod.beat_events and frame.new_beat:
                cmd_index = cmd_table.index(esc_getpos)
                h, l = cmd_index // 256, cmd_index % 256
                byte_stream += [0] * h + [l, 0x80 | (beat_index & 0x7f)]
                beat_index += 1
            # LSP runtime tempo changes carry a one-byte tracker BPM value.
            frame_bpm = self.mpt_mod.frames[i].tempo
            if not self.config.mod.mixer_experimental and frame_bpm != previous_bpm:
                if not 0 < frame_bpm <= 0xff:
                    raise ValueError(
                        f"SetBPM value {frame_bpm} at frame {i} does not fit in one byte"
                    )
                cmd_index = cmd_table.index(esc_setbpm)
                h, l = cmd_index // 256, cmd_index % 256
                byte_stream += [0] * h + [l, frame_bpm]
                previous_bpm = frame_bpm
            # Write command index, omitting index 0.
            # Each +256 index is a 00 byte, e.g. index 520 is 000008
            command = commands[i]
            cmd_index = cmd_table.index(command, 1)
            h, l = cmd_index // 256, cmd_index % 256
            byte_stream += [0] * h + [l]
            # Write channel commands. LSP expects reverse order (from 3 to 0).
            chans = frame.mir[::-1]
            byte_stream += [round(c.vol) for c in chans if c.vol is not None]
            word_stream += [round(c.per) for c in chans if c.per is not None]
            # If any instruments played, write instrument offset to word stream
            current_offset = -12
            for inst in [c.inst for c in chans if c.inst is not None]:
                inst_index = minsts.index(inst)
                offset = inst_index * 12 - current_offset
                word_stream.append(offset % 0x10000)
                current_offset += offset + 6
            if frame.event is not None:
                byte_stream.append(frame.event)
        # Store rewind special command
        cmd_index = cmd_table.index(esc_rewind)
        h, l = cmd_index // 256, cmd_index % 256
        byte_stream += [0] * h + [l]
        if self.config.mod.mixer_experimental:
            # BPM is instead number of CIA clock cycles per buffer
            size, _ = self._get_best_buffer_size()
            bpm = int(size / self.config.mod.mixer_rate * 1000 * 1000 / self.config.mod.cia_clock)
        # Build final LSP file
        lsp_file = LspFile(len(commands),
                           bpm,
                           esc_rewind,
                           esc_setbpm,
                           esc_getpos,
                           int(self.config.mod.getpos or self.config.mod.beat_events),
                           list(self.msamples.values()),
                           minsts,
                           cmd_table,
                           byte_loop,
                           word_loop,
                           word_stream,
                           byte_stream)
        # Get raw bytes and add final checksum header
        # NOTE: Checksum is calculated differently from original LSPConvert,
        # but it still serves the same purpose of giving LSP files a unique ID.
        bank_bytes = b''.join([msmp.data for msmp in self.msamples.values()])
        score_bytes = lsp_file.to_bytes()
        checksum = md5(bank_bytes + score_bytes, usedforsecurity=False).digest()[:4]
        if self.config.mod.mixer_experimental:
            size, count = self._get_best_buffer_size()
            checksum = size.to_bytes(2) + (count - 1).to_bytes(2)
        bank_bytes = checksum + bank_bytes
        score_bytes = b'LSP1' + checksum + score_bytes
        return bank_bytes, score_bytes

    def _get_best_buffer_size(self) -> tuple[int, int]:
        """ Returns the best buffer size and counter value to use that most
            closely results in the track's original chosen BPM."""
        # 21960Hz is 368 samples per frame, so max buffer of 184 will keep
        # SFX delay within one frame.
        # We take frame 1 to account for MODs that set tempo on frame 0
        smp_per_tick = self.mpt_mod.frames[1].samples_per_tick
        mx_smp_per_tick = smp_per_tick * self.config.mod.mixer_rate / mpt.MIX_RATE
        best_frac = math.inf
        best_quot = 0
        best_i = 0
        for i in range(92, 188, 4):
            div = mx_smp_per_tick / i
            frac = abs(div - round(div))
            if frac < best_frac:
                best_frac = frac
                best_i = i
                best_quot = round(div)
        # print(f'{int(mx_smp_per_tick)} -> {best_i} * {best_quot} ({best_i * best_quot})')
        return best_i, best_quot


class LspFile:
    """The final LSP score (.lsmusic) file."""
    version_major: int = 1
    version_minor: int = 25
    support_flags: int = 0 # %01 supports GetPos, %10 supports SetPos
    bpm: int
    esc_value_rewind: int
    esc_value_setbpm: int
    esc_value_getpos: int
    frame_count: int
    instrument_count: int
    instruments: list[LspInstrument]
    command_list_size: int
    commands: list[int]
    seq_count: int = 0
    seqs: list = []
    word_stream_size: int
    byte_stream_loop_pos: int = 0
    word_stream_loop_pos: int = 0
    word_stream: list[int]
    byte_stream: list[int]

    def __init__(self,
                 frame_count: int,
                 bpm: int,
                 rewind: int,
                 setbpm: int,
                 getpos: int,
                 support_flags: int,
                 samples: list[MSample],
                 minsts: list[MInstrument],
                 cmd_table: list[int],
                 byte_loop: int,
                 word_loop: int,
                 word_stream: list[int],
                 byte_stream: list[int]) -> None:
        self.bpm = bpm
        self.esc_value_rewind = rewind
        self.esc_value_setbpm = setbpm
        self.esc_value_getpos = getpos
        self.support_flags = support_flags
        self.frame_count = frame_count
        self.instrument_count = len(minsts)
        self.instruments = []
        # The 4 accounts for checksum header
        bank_offsets = dict(zip(
            [smp.id for smp in samples],
            (np.cumsum([0] + [len(smp.data) for smp in samples]) + 4).tolist()
            ))
        for minst in minsts:
            id = minst.sample.id
            smp = minst.sample
            sample_offset = bank_offsets[id] + minst.offset
            word_count = (len(smp.data) - minst.offset) // 2
            if smp.loop_end == 0:
                loop_offset = bank_offsets[id]
                loop_word_count = 1
            else:
                loop_offset = bank_offsets[id] + smp.loop_start
                loop_word_count = (smp.loop_end - smp.loop_start) // 2
            # If used in mixer, adjust offset to be after the MFX header
            if smp.mixer:
                sample_offset += 12
                loop_offset += 12
                word_count -= 6
                loop_word_count -= 6
                word_count = max(word_count, 1)
                loop_word_count = max(loop_word_count, 1)
            self.instruments.append(LspInstrument(sample_offset, word_count, loop_offset, loop_word_count))
        self.command_list_size = len(cmd_table)
        self.commands = cmd_table
        self.word_stream_size = len(word_stream) * 2
        self.byte_stream_loop_pos = byte_loop
        self.word_stream_loop_pos = word_loop
        self.word_stream = word_stream
        self.byte_stream = byte_stream

    def to_bytes(self) -> bytes:
        out = bytes()
        out += struct.pack('>BBHHHHHLH',
                           self.version_major,
                           self.version_minor,
                           self.support_flags,
                           self.bpm,
                           self.esc_value_rewind,
                           self.esc_value_setbpm,
                           self.esc_value_getpos,
                           self.frame_count,
                           self.instrument_count)
        out += b''.join(inst.to_bytes() for inst in self.instruments)
        out += self.command_list_size.to_bytes(2)
        out += b''.join(cmd.to_bytes(2) for cmd in self.commands)
        out += self.seq_count.to_bytes(2)
        out += b''.join(seq.to_bytes() for seq in self.seqs) # TODO
        out += struct.pack('>LLL',
                           self.word_stream_size,
                           self.byte_stream_loop_pos,
                           self.word_stream_loop_pos)
        out += b''.join(word.to_bytes(2) for word in self.word_stream)
        out += b''.join(byte.to_bytes(1) for byte in self.byte_stream)
        return out
