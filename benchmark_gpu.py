import time

import numpy as np
import pyopencl as cl


KERNEL_SOURCE = r"""
__kernel void evaluate_positions(
    __global const int *positions,
    __global const int *walls_remaining,
    __global float *scores
) {
    int i = get_global_id(0);

    int p0_y = positions[i * 4 + 1];
    int p1_y = positions[i * 4 + 3];

    int p0_walls = walls_remaining[i * 2];
    int p1_walls = walls_remaining[i * 2 + 1];

    // Player 0 bergerak menuju y = 0.
    int distance0 = p0_y;

    // Player 1 bergerak menuju y = 8.
    int distance1 = 8 - p1_y;

    float progress_score =
        (float)(distance1 - distance0) * 2.0f;

    float wall_score =
        (float)(p0_walls - p1_walls) * 0.25f;

    scores[i] = progress_score + wall_score;
}
"""


def evaluate_cpu(positions, walls_remaining):
    p0_y = positions[:, 1]
    p1_y = positions[:, 3]

    distance0 = p0_y
    distance1 = 8 - p1_y

    progress_score = (distance1 - distance0) * 2.0
    wall_score = (
        walls_remaining[:, 0] - walls_remaining[:, 1]
    ) * 0.25

    return (progress_score + wall_score).astype(np.float32)


def main():
    platforms = cl.get_platforms()

    devices = [
        device
        for platform in platforms
        for device in platform.get_devices()
        if device.type & cl.device_type.GPU
    ]

    if not devices:
        raise RuntimeError(
            "Tidak ada perangkat GPU OpenCL yang terdeteksi."
        )

    device = devices[0]

    print("GPU:", device.name)
    print("OpenCL version:", device.version)
    print("Compute units:", device.max_compute_units)

    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    program = cl.Program(context, KERNEL_SOURCE).build()

    kernel = program.evaluate_positions

    rng = np.random.default_rng(42)

    # Benchmark dengan beberapa ukuran batch.
    batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

    # Pemanasan untuk mengurangi efek inisialisasi awal.
    warmup_size = 1024

    warmup_positions = rng.integers(
        0, 9, size=(warmup_size, 4), dtype=np.int32
    )

    warmup_walls = rng.integers(
        0, 11, size=(warmup_size, 2), dtype=np.int32
    )

    warmup_scores = np.empty(warmup_size, dtype=np.float32)

    def run_gpu(positions, walls_remaining, scores):
        mf = cl.mem_flags

        pos_buffer = cl.Buffer(
            context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=positions,
        )

        walls_buffer = cl.Buffer(
            context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=walls_remaining,
        )

        scores_buffer = cl.Buffer(
            context,
            mf.WRITE_ONLY,
            scores.nbytes,
        )

        kernel(
            queue,
            (len(positions),),
            None,
            pos_buffer,
            walls_buffer,
            scores_buffer,
        )

        cl.enqueue_copy(
            queue,
            scores,
            scores_buffer,
        ).wait()

    run_gpu(warmup_positions, warmup_walls, warmup_scores)

    print()
    print(
        f"{'Batch':>10} | {'CPU (ms)':>12} | "
        f"{'GPU (ms)':>12} | {'Speedup':>10} | {'Hasil':>10}"
    )
    print("-" * 67)

    for size in batch_sizes:
        positions = rng.integers(
            0, 9, size=(size, 4), dtype=np.int32
        )

        walls_remaining = rng.integers(
            0, 11, size=(size, 2), dtype=np.int32
        )

        # Posisi ini hanya data sintetis untuk benchmark.
        # Belum mewakili state legal hasil permainan Quoridor.

        cpu_start = time.perf_counter()
        cpu_scores = evaluate_cpu(positions, walls_remaining)
        cpu_elapsed = time.perf_counter() - cpu_start

        gpu_scores = np.empty(size, dtype=np.float32)

        gpu_start = time.perf_counter()
        run_gpu(positions, walls_remaining, gpu_scores)
        gpu_elapsed = time.perf_counter() - gpu_start

        matches = np.allclose(
            cpu_scores,
            gpu_scores,
            rtol=1e-5,
            atol=1e-5,
        )

        speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed else float("inf")

        print(
            f"{size:10,d} | "
            f"{cpu_elapsed * 1000:12.3f} | "
            f"{gpu_elapsed * 1000:12.3f} | "
            f"{speedup:9.2f}x | "
            f"{'OK' if matches else 'MISMATCH':>10}"
        )

        if not matches:
            raise AssertionError(
                f"Hasil CPU dan GPU berbeda untuk batch {size}"
            )


if __name__ == "__main__":
    main()