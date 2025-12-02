import matplotlib.pyplot as plt
import numpy as np
from parse import parse_benchmark_file

def barplot_serial_vs_mpi(serial_heatmap, serial_filter, serial_hats,
                       mpi_heatmap, mpi_filter, mpi_hats, title, obarlabel, savepath):
    labels = ["Heatmap", "Background Filter", "HATS"]
    serial_vals = [serial_heatmap, serial_filter, serial_hats]
    mpi_vals = [mpi_heatmap, mpi_filter, mpi_hats]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, serial_vals, width, label="Serial")
    plt.bar(x + width/2, mpi_vals, width, label=obarlabel)

    plt.xticks(x, labels)
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def lineplot_time_vs_procs(times, procs, title, ylabel="Time (seconds)", xlabel="Number of Processes", savepath=None):
    plt.figure(figsize=(8,5))

    plt.plot(
        procs,
        times,
        marker="o",
        linewidth=2,
        markersize=8,
        label="MPI Time"
    )

    plt.xticks(procs, [str(p) for p in procs])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

    plt.close()

def lineplot_all_for_pixelareasize(
        heatmap_means,
        filter_means,
        hats_means,
        procs,
        title,
        savepath=None,
        ylabel="Time (seconds)",
        xlabel="Number of Processes"
    ):
    plt.figure(figsize=(8,5))

    plt.plot(
        procs,
        heatmap_means,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Heatmap"
    )

    plt.plot(
        procs,
        filter_means,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Background Filter"
    )

    plt.plot(
        procs,
        hats_means,
        marker="^",
        linewidth=2,
        markersize=8,
        label="HATS"
    )

    plt.xticks(procs, [str(p) for p in procs])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

    plt.close()


def perform_analysis(results):
    heatmap_64_small = results[("MPI Heatmap", "64")][1]
    heatmap_64_large = results[("MPI Heatmap", "64")][2]
    heatmap_32_small = results[("MPI Heatmap", "32(4x8)")][1]
    heatmap_32_large = results[("MPI Heatmap", "32(4x8)")][2]
    heatmap_16_small = results[("MPI Heatmap", "16")][1]
    heatmap_16_large = results[("MPI Heatmap", "16")][2]
    heatmap_4_small = results[("MPI Heatmap", "4")][1]
    heatmap_4_large = results[("MPI Heatmap", "4")][2]

    filt_64_small = results[("MPI Background Filtering", "64")][1]
    filt_64_large = results[("MPI Background Filtering", "64")][2]
    filt_32_small = results[("MPI Background Filtering", "32(4x8)")][1]
    filt_32_large = results[("MPI Background Filtering", "32(4x8)")][2]
    filt_16_small = results[("MPI Background Filtering", "16")][1]
    filt_16_large = results[("MPI Background Filtering", "16")][2]
    filt_4_small = results[("MPI Background Filtering", "4")][1]
    filt_4_large = results[("MPI Background Filtering", "4")][2]

    hats_64_small = results[("MPI HATS", "64")][1]
    hats_64_large = results[("MPI HATS", "64")][2]
    hats_32_small = results[("MPI HATS", "32(4x8)")][1]
    hats_32_large = results[("MPI HATS", "32(4x8)")][2]
    hats_16_small = results[("MPI HATS", "16")][1]
    hats_16_large = results[("MPI HATS", "16")][2]
    hats_4_small = results[("MPI HATS", "4")][1]
    hats_4_large = results[("MPI HATS", "4")][2]

    serial_heatmap_small = results[("Serial Heatmap", "serial")][1]
    serial_heatmap_large = results[("Serial Heatmap", "serial")][2]
    serial_filter_small = results[("Serial Background Filtering", "serial")][1]
    serial_filter_large = results[("Serial Background Filtering", "serial")][2]
    serial_hats_small = results[("Serial HATS descriptor", "serial")][1]
    serial_hats_large = results[("Serial HATS descriptor", "serial")][2]

    for key, tests in results.items():
        print(f"{key} Small:", tests[1])
        print(f"{key} Large:", tests[2])
        print()

    barplot_serial_vs_mpi(
        serial_heatmap=np.mean(serial_heatmap_small),
        serial_filter=np.mean(serial_filter_small),
        serial_hats=np.mean(serial_hats_small),
        mpi_heatmap=np.mean(heatmap_16_small),
        mpi_filter=np.mean(filt_16_small),
        mpi_hats=np.mean(hats_16_small),
        title="Serial vs MPI (16 Procs) on a 346x260 Pixel Area",
        obarlabel="MPI (16 Procs)",
        savepath="data_plots/barplot_serial_vs_mpi_small.svg"
        )
    
    barplot_serial_vs_mpi(
        serial_heatmap=np.mean(serial_heatmap_large),
        serial_filter=np.mean(serial_filter_large),
        serial_hats=np.mean(serial_hats_large),
        mpi_heatmap=np.mean(heatmap_16_large),
        mpi_filter=np.mean(filt_16_large),
        mpi_hats=np.mean(hats_16_large),
        title="Serial vs MPI (16 Procs) on a 1280x720 Pixel Area",
        obarlabel="MPI (16 Procs)",
        savepath="data_plots/barplot_serial_vs_mpi_large.svg"
        )
    
    mpi_heatmap_means_small = [np.mean(v) for v in [heatmap_4_small, heatmap_16_small, heatmap_32_small, heatmap_64_small]]
    mpi_heatmap_means_large = [np.mean(v) for v in [heatmap_4_large, heatmap_16_large, heatmap_32_large, heatmap_64_large]]
    mpi_filter_means_small = [np.mean(v) for v in [filt_4_small, filt_16_small, filt_32_small, filt_64_small]]
    mpi_filter_means_large = [np.mean(v) for v in [filt_4_large, filt_16_large, filt_32_large, filt_64_large]]
    mpi_hats_means_small = [np.mean(v) for v in [hats_4_small, hats_16_small, hats_32_small, hats_64_small]]
    mpi_hats_means_large = [np.mean(v) for v in [hats_4_large, hats_16_large, hats_32_large, hats_64_large]]

    lineplot_time_vs_procs(
        times=mpi_heatmap_means_small,
        procs=[4, 16, 32, 64],
        title="MPI Processing Performance (Heatmap, 346x260 Pixel Area)",
        savepath="data_plots/lineplot_time_vs_procs_heatmap_small.svg"
        )
    
    lineplot_time_vs_procs(
        times=mpi_heatmap_means_large,
        procs=[4, 16, 32, 64],
        title="MPI Processing Performance (Heatmap, 1280x720 Pixel Area)",
        savepath="data_plots/lineplot_time_vs_procs_heatmap_large.svg"
        )
    
    lineplot_time_vs_procs(
        times=mpi_filter_means_small,
        procs=[4, 16, 32, 64],
        title="MPI Processing Performance (Filter, 346x260 Pixel Area)",
        savepath="data_plots/lineplot_time_vs_procs_filter_small.svg"
        )
    
    lineplot_time_vs_procs(
        times=mpi_filter_means_large,
        procs=[4, 16, 32, 64],
        title="MPI Processing Performance (Filter, 1280x720 Pixel Area)",
        savepath="data_plots/lineplot_time_vs_procs_filter_large.svg"
        )
    
    lineplot_all_for_pixelareasize(
        heatmap_means=mpi_heatmap_means_small,
        filter_means=mpi_filter_means_small,
        hats_means=mpi_hats_means_small,
        procs=[4,16,32,64],
        title="MPI Performance Across Algorithms (346x260 Pixel Area)",
        savepath="data_plots/lineplot_all_small.svg"
    )

    lineplot_all_for_pixelareasize(
        heatmap_means=mpi_heatmap_means_large,
        filter_means=mpi_filter_means_large,
        hats_means=mpi_hats_means_large,
        procs=[4,16,32,64],
        title="MPI Performance Across Algorithms (1280x720 Pixel Area)",
        savepath="data_plots/lineplot_all_large.svg"
    )
    
def main():
    r1_path = "DV_MPI_TEST_149984.out"
    r1 = parse_benchmark_file(r1_path)

    r2_path = "DV_MPI_TEST_149985.out"
    r2 = parse_benchmark_file(r2_path)

    perform_analysis(r1)


if __name__ == "__main__":
    main()
