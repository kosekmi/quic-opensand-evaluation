#!/usr/bin/python3

import csv
import json
from collections import defaultdict
import math
import os.path
import sys
import getopt

server_ip = "10.10.0.3"
client_ip = "10.10.2.1"


# source: http://code.activestate.com/recipes/511478/
def percentile(N, percent, key=lambda x: x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1


class StatsList:
    def __init__(self):
        self.items = list()

    def __len__(self):
        return len(self.items)

    def add_packets(self, first, second):
        self.items.append(float(second.sniff_timestamp) - float(first.sniff_timestamp))

    def add_val(self, val):
        self.items.append(val)

    def cnt(self):
        return len(self.items)

    def mean(self):
        if len(self.items) == 0:
            return 0
        return sum(self.items) / len(self.items)

    def max(self):
        if len(self.items) == 0:
            return 0
        return max(self.items)

    def min(self):
        if len(self.items) == 0:
            return 0
        return min(self.items)

    def p95(self):
        if len(self.items) == 0:
            return 0
        return percentile(sorted(self.items), 0.95)


def load_json_file(path):
    if not os.path.isfile(path):
        return None

    with open(path, 'r') as json_file:
        try:
            return json.load(json_file)
        except json.JSONDecodeError as e:
            json_file.seek(0)
            json_str = json_file.read(e.pos)
            return json.loads(json_str)


def curl_established_time(log):
    stats = StatsList()
    with open(log) as logfile:
        for line in logfile:
            measurement = line.split(" ")[0].split("=")
            if len(measurement) != 2 or measurement[0] != "established":
                continue
            conn_est = float(measurement[1].replace(",", "."))
            stats.add_val(conn_est)
    return stats


def curl_ttfb(log):
    stats = StatsList()
    with open(log) as logfile:
        for line in logfile:
            measurement = line.split(" ")[1].split("=")
            if len(measurement) != 2 or measurement[0] != "ttfb":
                continue
            ttfb = float(measurement[1].replace(",", "."))
            stats.add_val(ttfb)
    return stats


def tcp_cwnd_evo(path):
    evo = defaultdict(StatsList)

    for i in range(11):
        result_file = os.path.join(path, f"{i}_tcp_cwnd_evo.json")
        if not os.path.isfile(result_file):
            print(f"{result_file} not found, skipping")
            continue

        iperf = load_json_file(result_file)
        if iperf is None:
            continue

        for idx, interval in enumerate(iperf['intervals']):
            snd_cwnd = interval['streams'][0]['snd_cwnd']
            if snd_cwnd > 0:
                evo[idx].add_val(snd_cwnd)

    if len(evo) > 0:
        print(f"tcp_cwnd_evo: {max([x.cnt() for x in evo.values()])} items")
    return evo


def tcp_goodput(path):
    goodput = defaultdict(StatsList)

    for i in range(11):
        result_file = os.path.join(path, f"{i}_tcp_goodput.json")
        if not os.path.isfile(result_file):
            print(f"{result_file} not found, skipping")
            continue

        iperf = load_json_file(result_file)
        if iperf is None:
            continue

        for idx, interval in enumerate(iperf['intervals']):
            bps = interval['streams'][0]['bits_per_second']
            if bps > 0:
                goodput[idx].add_val(bps)

    if len(goodput) > 0:
        print(f"tcp_goodput: {max([x.cnt() for x in goodput.values()])} items")
    return goodput


def qperf_conn_time(logs_path):
    stats = StatsList()
    path = os.path.join(logs_path, f"quic_ttfb.txt")
    if not os.path.isfile(path):
        print(f"quicly conn time error: {path} does not exist")
        return

    with open(path) as logfile:
        for line in logfile:
            line = line.strip()
            if not line.startswith("connection establishment time:"):
                continue

            conn_est_time = int(line.split(" ")[-1][:-2]) / 1000
            stats.add_val(conn_est_time)

    print(f"qperf_conn_time: {stats.cnt()} items")
    return stats


def qperf_time_to_first_byte(logs_path):
    stats = StatsList()
    path = os.path.join(logs_path, f"quic_ttfb.txt")
    if not os.path.isfile(path):
        print(f"quicly time to first byte error: {path} does not exist")
        return

    with open(path) as logfile:
        for line in logfile:
            line = line.strip()
            if not line.startswith("time to first byte:"):
                continue
            conn_est_time = int(line.split(" ")[-1][:-2]) / 1000
            stats.add_val(conn_est_time)

    print(f"qperf_time_to_first_byte: {stats.cnt()} items")
    return stats


def qperf_cwnd_evo(log_path):
    evo_stats = defaultdict(StatsList)
    for i in range(13):
        path = os.path.join(log_path, f"{i}_quic_cwnd_evo.txt")
        if not os.path.isfile(path):
            continue

        with open(path) as logfile:
            for line in logfile:
                line = line.strip()
                if not line.startswith("connection 0 second"):
                    continue
                second = int(line.split(" ")[3])
                cwnd = int(line.split(" ")[6])
                evo_stats[second].add_val(cwnd)

    print(f"quic_cwnd_evo: {max(x.cnt() for x in evo_stats.values())} items")
    return evo_stats


def qperf_goodput(log_path):
    goodput_stats = defaultdict(StatsList)
    for i in range(13):
        path = os.path.join(log_path, f"{i}_quic_goodput.txt")
        if not os.path.isfile(path):
            continue

        with open(path) as logfile:
            for line in logfile:
                line = line.strip()
                if not line.startswith("second"):
                    continue
                second = int(line.split(" ")[1][:-1])
                bytes_received = int(line.split(" ")[4][1:])
                goodput_stats[second].add_val(bytes_received * 8)

    print(f"quic_goodput: {max(x.cnt() for x in goodput_stats.values())} items")
    return goodput_stats


def measure_folders(root_folder):
    for delay in ["LEO", "MEO", "GEO"]:
        for rate in ["1mbit", "10mbit", "100mbit"]:
            for loss in ["0.01", "0.1", "1", "5"]:
                for queue in ["1", "2", "5", "10"]:
                    for pep in ["none", "bbr", "hybla", "fec"]:
                        pep_text = f"_{pep}" if pep != "none" else ""
                        measure_folder = os.path.join(root_folder, f"{delay}_r{rate}_l{loss}_q{queue}{pep_text}")

                        if os.path.isdir(measure_folder):
                            yield delay, rate, loss + '%', queue, pep, measure_folder


def parse(in_dir="~/measure", out_dir="."):
    tcp_conn_times = list()
    tcp_ttfb = list()
    tcp_cwnd_evos = list()
    tcp_goodputs = list()

    tls_conn_times = list()
    tls_ttfb = list()

    quic_conn_times = list()
    quic_ttfb = list()
    quic_cwnd_evos = list()
    quic_goodputs = list()

    quic_fec_conn_times = list()
    quic_fec_ttfb = list()
    quic_fec_cwnd_evos = list()
    quic_fec_goodputs = list()

    for delay, rate, loss, queue, pep, folder in measure_folders(in_dir):
        print(f"\ndelay={delay} rate={rate} loss={loss} queue={queue} pep={pep}")

        # tcp & tls -----------------------------------------------------------------------------------
        result_file = os.path.join(folder, "tcp_conn_est.txt")
        if os.path.isfile(result_file):
            s = curl_established_time(result_file)
            print(f"tcp_conn_time: {s.cnt()} items")
            tcp_conn_times.append([delay, rate, loss, queue, pep, s.mean(), s.min(), s.max(), s.p95()])
        else:
            print(f"{result_file} not found, skipping")

        # again???
        result_file = os.path.join(folder, "tls_conn_est.txt")
        if os.path.isfile(result_file):
            s = curl_established_time(result_file)
            print(f"tls_conn_time: {s.cnt()} items")
            tls_conn_times.append([delay, rate, loss, queue, pep, s.mean(), s.min(), s.max(), s.p95()])
        else:
            print(f"{result_file} not found, skipping")

        result_file = os.path.join(folder, "tcp_conn_est.txt")
        if os.path.isfile(result_file):
            s = curl_ttfb(result_file)
            print(f"tcp_ttfb: {s.cnt()} items")
            tcp_ttfb.append([delay, rate, loss, queue, pep, s.mean(), s.min(), s.max(), s.p95()])
        else:
            print(f"{result_file} not found, skipping")

        # again???
        result_file = os.path.join(folder, "tls_conn_est.txt")
        if os.path.isfile(result_file):
            s = curl_ttfb(result_file)
            print(f"tls_ttfb: {s.cnt()} items")
            tls_ttfb.append([delay, rate, loss, queue, pep, s.mean(), s.min(), s.max(), s.p95()])
        else:
            print(f"{result_file} not found, skipping")

        for t, cwnd in sorted(tcp_cwnd_evo(folder).items(), key=lambda x: x[0]):
            tcp_cwnd_evos.append([delay, rate, loss, queue, pep, t,
                                  int(cwnd.mean()), int(cwnd.min()), int(cwnd.max()), int(cwnd.p95())])

        for t, goodput in sorted(tcp_goodput(folder).items(), key=lambda x: x[0]):
            tcp_goodputs.append(([delay, rate, loss, queue, pep, t,
                                  int(goodput.mean()), int(goodput.min()), int(goodput.max()), int(goodput.p95())]))

        if pep == "none":
            # quic ----------------------------------------------------------------------------------------
            s = qperf_conn_time(folder)
            quic_conn_times.append([delay, rate, loss, queue, s.mean(), s.min(), s.max(), s.p95()])

            s = qperf_time_to_first_byte(folder)
            quic_ttfb.append([delay, rate, loss, queue, s.mean(), s.min(), s.max(), s.p95()])

            for t, cwnd in sorted(qperf_cwnd_evo(folder).items(), key=lambda x: x[0]):
                quic_cwnd_evos.append([delay, rate, loss, queue, t,
                                       int(cwnd.mean()), int(cwnd.min()), int(cwnd.max()), int(cwnd.p95())])

            for t, goodput in sorted(qperf_goodput(folder).items(), key=lambda x: x[0]):
                quic_goodputs.append([delay, rate, loss, queue, t,
                                      int(goodput.mean()), int(goodput.min()), int(goodput.max()), int(goodput.p95())])

        elif pep == "fec":
            # quic over fec tunnel ------------------------------------------------------------------------
            s = qperf_conn_time(folder)
            quic_fec_conn_times.append([delay, rate, loss, queue, s.mean(), s.min(), s.max(), s.p95()])

            s = qperf_time_to_first_byte(folder)
            quic_fec_ttfb.append([delay, rate, loss, queue, s.mean(), s.min(), s.max(), s.p95()])

            for t, cwnd in sorted(qperf_cwnd_evo(folder).items(), key=lambda x: x[0]):
                quic_fec_cwnd_evos.append([delay, rate, loss, queue, t,
                                           int(cwnd.mean()), int(cwnd.min()), int(cwnd.max()), int(cwnd.p95())])

            for t, goodput in sorted(qperf_goodput(folder).items(), key=lambda x: x[0]):
                quic_fec_goodputs.append([delay, rate, loss, queue, t,
                                          int(goodput.mean()), int(goodput.min()), int(goodput.max()),
                                          int(goodput.p95())])

    # Write results
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_dir):
        print(f"{out_dir} is not a directory, cannot write parsed results")
        return

    with open(os.path.join(out_dir, "tcp_connection_establishment.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "mean", "min", "max", "p95"])
        writer.writerows(tcp_conn_times)

    with open(os.path.join(out_dir, "tls_connection_establishment.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "mean", "min", "max", "p95"])
        writer.writerows(tls_conn_times)

    with open(os.path.join(out_dir, "tcp_time_to_first_byte.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "mean", "min", "max", "p95"])
        writer.writerows(tcp_ttfb)

    with open(os.path.join(out_dir, "tls_time_to_first_byte.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "mean", "min", "max", "p95"])
        writer.writerows(tls_ttfb)

    with open(os.path.join(out_dir, "tcp_cwnd_evolution.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "t", "mean", "min", "max", "p95"])
        writer.writerows(tcp_cwnd_evos)

    with open(os.path.join(out_dir, "tcp_goodput.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "pep", "t", "mean", "min", "max", "p95"])
        writer.writerows(tcp_goodputs)

    with open(os.path.join(out_dir, "quic_connection_establishment.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "mean", "min", "max", "p95"])
        writer.writerows(quic_conn_times)

    with open(os.path.join(out_dir, "quic_time_to_first_byte.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "mean", "min", "max", "p95"])
        writer.writerows(quic_ttfb)

    with open(os.path.join(out_dir, "quic_cwnd_evolution.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "t", "mean", "min", "max", "p95"])
        writer.writerows(quic_cwnd_evos)

    with open(os.path.join(out_dir, "quic_goodput.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "t", "mean", "min", "max", "p95"])
        writer.writerows(quic_goodputs)

    with open(os.path.join(out_dir, "quic_fec_connection_establishment.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "mean", "min", "max", "p95"])
        writer.writerows(quic_fec_conn_times)

    with open(os.path.join(out_dir, "quic_fec_time_to_first_byte.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "mean", "min", "max", "p95"])
        writer.writerows(quic_fec_ttfb)

    with open(os.path.join(out_dir, "quic_fec_cwnd_evolution.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "t", "mean", "min", "max", "p95"])
        writer.writerows(quic_fec_cwnd_evos)

    with open(os.path.join(out_dir, "quic_fec_goodput.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delay", "rate", "loss", "queue", "t", "mean", "min", "max", "p95"])
        writer.writerows(quic_fec_goodputs)

    print("done")


def main(argv):
    in_dir = "~/measure"
    out_dir = "."

    try:
        opts, args = getopt.getopt(argv, "i:o:", ["input=", "output="])
    except getopt.GetoptError:
        print("parse.py -i <inputdir> -o <outputdir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            in_dir = arg
        elif opt in ("-o", "--output"):
            out_dir = arg

    parse(in_dir, out_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
