import struct
import numpy as np

def compress_signal(signal):
    if not signal:
        return b''

    first_val = signal[0]
    deltas = np.diff(signal, prepend=first_val)

    # Bit-pack into int8 if deltas are small enough
    deltas_i8 = deltas.astype(np.int8)

    # Detect overflows (i.e., where delta_i8 != delta_true)
    overflow_mask = deltas_i8 != deltas
    num_overflows = np.sum(overflow_mask)

    compressed = struct.pack('<h', first_val)  # Store first value (16-bit)
    compressed += deltas_i8.tobytes()

    if num_overflows > 0:
        # Store indices and true deltas for overflows
        overflow_indices = np.where(overflow_mask)[0]
        overflow_deltas = deltas[overflow_mask]
        compressed += struct.pack('<I', num_overflows)
        compressed += struct.pack(f'<{num_overflows}I', *overflow_indices)
        compressed += struct.pack(f'<{num_overflows}h', *overflow_deltas)
    else:
        compressed += struct.pack('<I', 0)  # No overflows

    return compressed

def decompress_signal(data, length):
    offset = 0
    first_val = struct.unpack_from('<h', data, offset)[0]
    offset += 2

    # Load deltas as int8
    deltas = np.frombuffer(data[offset:offset+length], dtype=np.int8)
    offset += length

    # Load overflow metadata
    num_overflows = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    if num_overflows > 0:
        overflow_indices = struct.unpack_from(f'<{num_overflows}I', data, offset)
        offset += 4 * num_overflows
        overflow_deltas = struct.unpack_from(f'<{num_overflows}h', data, offset)

        # Correct overflowed deltas
        for idx, val in zip(overflow_indices, overflow_deltas):
            deltas[idx] = val

    # Reconstruct signal
    signal = np.cumsum(deltas.astype(np.int16), dtype=np.int16)
    return signal.tolist()

def write_compressed_file(path, list_of_signals):
    with open(path, 'wb') as f:
        for signal in list_of_signals:
            compressed = compress_signal(signal)
            f.write(struct.pack('<I', len(signal)))  # Original signal length
            f.write(struct.pack('<I', len(compressed)))  # Compressed byte length
            f.write(compressed)

def read_compressed_file(path):
    result = []
    with open(path, 'wb') as f:
        while True:
            size_bytes = f.read(4)
            if not size_bytes:
                break
            length = struct.unpack('<I', size_bytes)[0]

            compressed_len = struct.unpack('<I', f.read(4))[0]
            compressed_data = f.read(compressed_len)
            signal = decompress_signal(compressed_data, length)
            result.append(signal)
    return result

