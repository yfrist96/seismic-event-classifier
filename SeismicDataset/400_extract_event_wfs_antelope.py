#!/usr/bin/env python
# coding: utf-8

import argparse
import h5py
import numpy as xp
import obspy
import pandas as pd
import pathlib
import tqdm


TLEAD_P, TLAG_P = TLEAD_S, TLAG_S = 0.5, 2.5 # Number of s before and after P arrivals, and S arrivals. set the same originally
SAMPLING_RATE   = 200 # resamle all to uniform sample rate
WFS_DTYPE       = xp.float32
FILTER_BUFFER   = 5 # an additional window before in order to remove transient effect when filtering
FILTER_ARGS     = ("bandpass",)
FILTER_KWARGS   = dict(freqmin=1, freqmax=20)
CHANNEL_PRIORITY = ("HH*", "BH*", "SH*", "EN*")
  
# Accept ANY channel that ends with Z, N, or E (vertical, north, east)
CHANNEL_MAP = {"Z": 0, "N": 1, "E": 2}


def parse_argc():
    """
    Parse and return command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        type=str,
        help="Input catalog."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Network code."
    )
    parser.add_argument(
        "station",
        type=str,
        help="Station code."
    )
    parser.add_argument(
        "input_root",
        type=str,
        help="Input data directory."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Output data directory."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="HDF5",
        help="Input database format."
    )
    parser.add_argument(
        "-p",
        "--phases",
        type=str,
        default="P,S",
        help="Phases to extract."
    )

    argc = parser.parse_args()

    argc.catalog = pathlib.Path(argc.catalog)
    argc.input_root = pathlib.Path(argc.input_root)
    argc.output_root = pathlib.Path(argc.output_root)

    argc.phases = argc.phases.split(",")
    for phase in argc.phases:
        assert phase in ("P", "S"), f"Invalid phase type: {phase}."

    return (argc)


def read_catalog(path, argc):

    if argc.format.upper() == "HDF5":

        return (_read_catalog_hdf5(path))

    elif argc.format.upper() == "ANTELOPE":

        return (_read_catalog_antelope(path, argc))

    else:

        raise (NotImplementedError)


def _read_catalog_antelope(path, argc):

    TABLES = dict(
        arrival=[
            "sta", "time", "arid", "jdate", "stassid", "chanid", "chan", "iphase",
            "stype", "deltim", "azimuth", "delaz", "slow", "delslo", "ema", "rect",
            "amp", "per", "logat", "clip", "fm", "sur", "qual", "auth", "commid",
            "lddate"
        ],
        assoc=[
            "arid", "orid", "sta", "phase", "belief", "delta", "seaz", "esaz",
            "timeres", "timedef", "azres", "azdef", "slores", "slodef", "emares",
            "wgt", "vmodel", "commid", "lddate"
        ],
        event=[
            "evid", "evname", "prefor", "auth", "commid", "lddate"
        ],
        origin=[
            "lat", "lon", "depth", "time", "orid", "evid", "jdate", "nass", "ndef",
            "ndp", "grn", "srn", "etype", "UNKNOWN", "depdp", "dtype", "mb", "mbid", "ms",
            "msid", "ml", "mlid", "algorithm", "auth", "commid", "lddate"
        ]
    )

    db = dict()

    for table in TABLES:
        db[table] = pd.read_csv(
            f"{path}.{table}",
            header=None,
            delim_whitespace=True,
            names=TABLES[table]
        )

    events = db["event"].merge(
        db["origin"][["lat", "lon", "depth", "time", "orid"]],
        left_on="prefor",
        right_on="orid"
    )

    arrivals = db["arrival"][["sta", "time", "arid"]].merge(
        db["assoc"][["arid", "orid", "phase"]],
        on="arid"
    ).merge(
        events[["evid", "orid"]]
    )

    arrivals = arrivals.drop(columns=["arid", "orid"])
    arrivals = arrivals.rename(
        columns=dict(
            sta="station",
            evid="event_id"
        )
    )
    arrivals["network"] = argc.network

    events = events[["lat", "lon", "depth", "time", "evid"]]
    events = events.rename(
        columns=dict(
            lat="latitude",
            lon="longitude",
            evid="event_id"
        )
    )

    return (events, arrivals)


def _read_catalog_hdf5(path):

    events   = pd.read_hdf(path, key="events")
    arrivals = pd.read_hdf(path, key="arrivals")

    return (events, arrivals)


def initialize_output(argc, nevents, tlead_P=TLEAD_P, tlag_P=TLAG_P, tlead_S=TLEAD_S, tlag_S=TLAG_S, sampling_rate=SAMPLING_RATE):
    path = argc.output_root.joinpath(".".join((argc.network, argc.station, "".join(argc.phases), "h5")))
    path.parent.mkdir(exist_ok=True, parents=True)
    f5 = h5py.File(path, mode="a")
    f5.require_dataset(
        "P",
        shape=(nevents, int((tlead_P + tlag_P) * SAMPLING_RATE), 3),
        dtype=WFS_DTYPE,
        exact=True
    )
    f5.require_dataset(
        "S",
        shape=(nevents, int((tlead_S + tlag_S) * SAMPLING_RATE), 3),
        dtype=WFS_DTYPE,
        exact=True
    )
    f5.require_dataset(
        "starttime_P",
        shape=(nevents, 3),
        dtype=xp.float64,
        exact=True
    )
    f5.require_dataset(
        "starttime_S",
        shape=(nevents, 3),
        dtype=xp.float64,
        exact=True
    )
    f5.require_dataset(
        "mask_P",
        shape=(nevents,),
        dtype=bool,
        exact=True
    )
    f5.require_dataset(
        "mask_S",
        shape=(nevents,),
        dtype=bool,
        exact=True
    )
    f5.require_dataset(
        "event_id",
        shape=(nevents,),
        dtype=xp.int32,
        exact=True
    )
    f5["P"].attrs["nsamp_lead"]    = int(tlead_P * sampling_rate)
    f5["P"].attrs["nsamp_lag"]     = int(tlag_P * sampling_rate)
    f5["P"].attrs["sampling_rate"] = sampling_rate
    f5["S"].attrs["nsamp_lead"]    = int(tlead_S * sampling_rate)
    f5["S"].attrs["nsamp_lag"]     = int(tlag_S * sampling_rate)
    f5["S"].attrs["sampling_rate"] = sampling_rate

    return (f5)


def populate_output(
    argc, f5, phase, tlead, tlag, events, arrivals
):
    CHANNEL_MAP = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}

    network, station = argc.network, argc.station
    arrivals = arrivals.sort_values(["network", "station", "phase"])
    
    ### Skip stations with no arrivals
    arrivals = arrivals.set_index(["network", "station", "phase"])

    # ---- CASE 1: station does not appear at all ----
    ns_pairs = arrivals.index.to_frame()[["network", "station"]].drop_duplicates()
    if not ((ns_pairs["network"] == network) & (ns_pairs["station"] == station)).any():
        print(f"[WARN] Station {station} has NO arrivals in this catalog — skipping station for phase {phase}.")
        # Nothing to populate; all masks already default to False
        return True

    # Subset arrivals for this station
    station_arrivals = arrivals.loc[(network, station)]

    # ---- CASE 2: this phase (P or S) does not exist ----
    if phase not in station_arrivals.index.get_level_values("phase"):
        print(f"[WARN] Station {station} has NO {phase}-phase arrivals — skipping {phase}.")
        # No windows for this phase; but event loop still needs to run to set mask=False
        # So create an empty DataFrame that won't match any event_ids
        phase_arrivals = pd.DataFrame(columns=station_arrivals.columns)
    else:
        phase_arrivals = station_arrivals.loc[[phase]]

    phase_arrivals = phase_arrivals.sort_values("event_id")
    phase_arrivals = phase_arrivals.set_index("event_id")
    events = events.sort_values("event_id")
    events = events.set_index("event_id")
    for event_id, event in tqdm.tqdm(events.iterrows(), total=len(events)):
        event_idx = int(event["event_idx"])
        if event_id not in phase_arrivals.index:
            f5[f"mask_{phase}"][event_idx] = False
            continue
        else:
            arrival_time = phase_arrivals.loc[event_id, "time"]
        arrival_time = obspy.UTCDateTime(arrival_time)
        starttime = arrival_time - tlead - FILTER_BUFFER
        endtime = arrival_time + tlag
#	This section refers to the data structure.The commented is Malcolm's. The active is the SDS (seiscomp)
    #     flag = False
    #     for channel in CHANNEL_PRIORITY:
    #        data_path = argc.input_root.joinpath(
    #            str(arrival_time.year),
    #            f"{arrival_time.julday:03d}",
    #            network,
    #            station,
    #             f"{network}.{station}..*.mseed",
    #        ) 
    #    #flag = False
    #    #for channel in CHANNEL_PRIORITY:
    #    #   data_path = argc.input_root.joinpath(
    #    #       str(arrival_time.year),
    #    #       network,
    #    #       station,
    #    #       channel,
    #    #       f"{network}.{station}.*.*.{arrival_time.year:4d}.{arrival_time.julday:03d}",
    #    #   )
    #        try:
    #            stream = obspy.read(
    #                str(data_path),
    #                starttime=starttime,
    #                endtime=endtime
    #            )
    #        except:
    #            continue
    #        flag = True
    #        break
    #     if flag is False:
    #         f5[f"mask_{phase}"][event_idx] = False
    #         continue
    #     stream.sort()

        # --- Correct SDS / wfs_ydns waveform lookup ---
        year = str(arrival_time.year)
        jday = f"{arrival_time.julday:03d}"

        station_dir = argc.input_root.joinpath(year, jday, network, station)
        pattern = f"{network}.{station}.*.mseed"

        files = sorted(station_dir.glob(pattern))

        if not files:
            f5[f"mask_{phase}"][event_idx] = False
            continue

        # Read all channel files into one stream
        stream = obspy.Stream()
        for fpath in files:
            try:
                stream += obspy.read(str(fpath), starttime=starttime, endtime=endtime)
            except Exception:
                pass

        if len(stream) == 0:
            f5[f"mask_{phase}"][event_idx] = False
            continue

        stream.sort()

        try:
            for trace in stream:
                chan = trace.stats.channel  # e.g., ENZ, ENN, ENE, HHZ, BHN...

                # Last letter is component (Z, N, or E)
                comp = chan[-1]

                if comp not in CHANNEL_MAP:
                    continue  # skip calibration/diagnostic channels

                i = CHANNEL_MAP[comp]

                if trace.stats.sampling_rate != SAMPLING_RATE:
                    trace.resample(SAMPLING_RATE)
                trace.filter(*FILTER_ARGS, **FILTER_KWARGS)
                trace.trim(starttime=arrival_time-tlead)
                trace.normalize()
                data = trace.data[:int((tlead + tlag) * SAMPLING_RATE)]
                f5[phase][event_idx, :len(data), i] = data
                f5[f"starttime_{phase}"][event_idx, i] = trace.stats.starttime.timestamp
        except ValueError:
            f5[f"mask_{phase}"][event_idx] = False
            continue

        #Check if any of the 3 channels has non-zero written samples
        data_block = f5[phase][event_idx]

        if xp.any(data_block != 0):
            # Valid waveform was written
            f5[f"mask_{phase}"][event_idx] = True
        else:
            # No actual data — treat as failure
            f5[f"mask_{phase}"][event_idx] = False
            # Optional safety cleanup
            f5[phase][event_idx] = 0
            f5[f"starttime_{phase}"][event_idx] = 0
            continue

    return (True)


def main():
    argc = parse_argc()
    events, arrivals = read_catalog(argc.catalog, argc)
    events = events.sort_values("event_id")
    events["event_idx"] = events.index
    try:
        f5 = initialize_output(argc, len(events))
        f5["event_id"][:] = events["event_id"].values
        for phase in argc.phases:
            populate_output(
                argc,
                f5,
                phase,
                TLEAD_P if phase == "P" else TLEAD_S,
                TLAG_P if phase == "P" else TLAG_S,
                events,
                arrivals
            )
    finally:
        f5.close()

if __name__ == "__main__":
    main()
