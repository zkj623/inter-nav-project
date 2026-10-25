"""Inspect bounded CDR video envelopes; never decode images or infer ROS time.

The installed/official Go2FrontVideoData declares uint64 + three byte sequences.
The audited robot instead sends uint64 + uint32 stream tag + one byte sequence
under that same DDS type name. Keep this compatibility local to diagnostics;
do not replace the system message definition based on inferred field names.
"""
import struct


MAX_SERIALIZED_BYTES = 16 * 1024 * 1024


def describe_front_video(data):
    """Validate the entire envelope and return metadata without video payloads."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError('Front video requires serialized CDR bytes')
    data = memoryview(data)
    if not 12 <= len(data) <= MAX_SERIALIZED_BYTES:
        raise ValueError('Front video serialized size is outside supported bounds')
    encapsulation = bytes(data[:2])
    if encapsulation not in (b'\x00\x00', b'\x00\x01'):
        raise ValueError('Unsupported front video CDR representation')
    options = int.from_bytes(data[2:4], 'big')
    if options & ~3:
        raise ValueError('Unsupported front video CDR options')
    endian = '<' if encapsulation == b'\x00\x01' else '>'
    end = len(data) - options
    if end < 12:
        raise ValueError('Truncated front video timestamp')
    frame = struct.unpack_from(endian+'Q', data, 4)[0]
    common = {'time_frame': frame, 'serialized_bytes': len(data), 'cdr_padding_bytes': options}

    # Observed wire layout. A tag is metadata, not the length of video720p.
    if end >= 20:
        tag, size = struct.unpack_from(endian+'II', data, 12)
        if tag in (180, 360, 720) and size > 0 and 20+size == end:
            prefix = bytes(data[20:min(end, 24)])
            if not prefix.startswith((b'\x00\x00\x01', b'\x00\x00\x00\x01')):
                raise ValueError('Tagged front video lacks an Annex B start code')
            return dict(common, wire_layout='tagged_single_stream',
                        declared_schema_matches=False, stream_tag=tag,
                        encoded_bytes=size, payload_framing='annex_b')

    # Also support the declared legacy schema. Bound each length before slicing
    # or allocating; malformed lengths must not reach native sequence allocation.
    pos = 12
    lengths = {}
    for name in ('video720p', 'video360p', 'video180p'):
        pos = 4 + ((pos-4+3)//4)*4
        if pos+4 > end:
            raise ValueError(f'Front video envelope mismatch: missing {name} length')
        size = struct.unpack_from(endian+'I', data, pos)[0]
        pos += 4
        if size > end-pos:
            raise ValueError(f'Front video envelope mismatch: {name} exceeds remaining bytes')
        lengths[name] = size
        pos += size
    if pos != end:
        raise ValueError('Front video envelope mismatch: unexplained trailing bytes')
    return dict(common, wire_layout='declared_three_streams',
                declared_schema_matches=True, encoded_bytes=lengths)
