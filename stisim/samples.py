import zipfile
import io
import sciris as sc
import pandas as pd
from collections import namedtuple, defaultdict
import numpy as np
import copy
from pathlib import Path

# Hierarchy
# - result: parameters, seed, beta
# - samples: collection of results with same parameters but different seeds
# - dataset: collection of samples (with different parameters)

__all__ = ['Dataset', 'Samples']

class Dataset:
    def __init__(self, folder=None, results=None, *args, **kwargs):
        # Note that results are not deep copied, to save memory
        if folder is not None:
            self.results = sc.odict()
            for file in sorted(Path(folder).iterdir()):
                if file.suffix == ".zip":
                    res = Samples(file, *args, **kwargs)
                    self.results[res.identifier] = res
        elif results is not None:
            self.results = results
        else:
            raise Exception("Must provide either folder or results to create Dataset")

    @property
    def ids(self):
        """
        Return dictionary of parameters across results
        """
        ids = defaultdict(set)
        for res in self.results.values():
            for k, v in res.id.items():
                ids[k].add(v)
        return {k: sorted(v) for k, v in ids.items()}

    def __repr__(self):
        s = "<Dataset:\n"
        for k, v in self.ids.items():
            s += f"\t'{k}':{v}\n"
        s += ">"
        return s

    def __getitem__(self, item):
        # Dict-like access for results
        return self.results[item]

    def __iter__(self):
        for result in self.results.values():
            yield result

    def __len__(self):
        return len(self.results)

    def __bool__(self):
        return len(self.results) > 0

    def filter(self, **kwargs):
        # Return results matching particular ids
        # kwargs: key-value pairs, that should be present in Sample.id, to filter results on
        # Can specify a list/set of values to match multiple items
        results = sc.odict()
        for res in self:
            for k, v in kwargs.items():
                if isinstance(v, list) or isinstance(v, set):
                    if res.id[k] not in v:
                        break
                elif res.id[k] != v:
                    break  # This result does not match, skip it
            else:
                results[res.identifier] = res

        return Dataset(results=results)

    def get(self, **kwargs):
        # Retrieve a single result from a filter operation
        # e.g. `res = Dataset.get(scenario='foo')` instead of
        # `res = Dataset.filter(scenario='foo')[0]` instead of
        # assuming that the arguments result in only 1 result being selected
        ds = self.filter(**kwargs)
        if len(ds) == 0:
            raise KeyError("No matching results were found")
        elif len(ds) > 1:
            raise KeyError("Multiple results matched the requested criteria")
        else:
            return ds[0]


class Samples:
    """
    Stores CSV outputs and summary dataframes

    To construct, use ``Samples.new()``. To read an existing one, use ``Samples(fname)``.
    The sample files are just ZIP archives with plain text CSV and TXT files so they
    can be easily accessed externally as well.
    """

    def __init__(self, fname, memory_buffer=True, preload=False):
        """

        Args:
            fname: Name of zip file to load
            memory_buffer: Load the file into memory. This avoids locking the file on disk and
                           improves performance when loading random parts of the file. This option can be
                           disabled in scripts to reduce memory requirements and improve performance if
                           not accessing the raw dataframes
            preload: Load all of the dataframes into memory during construction. This slows down initialization
                     but subsequent access to the dataframes will be fast - otherwise, there will be overhead incurred
                     the first time that the dataframes are used. Preloading is useful for interactive scripts where
                     it is known that the dataframes will be required. It should not be used if only the summary
                     is needed.
        """

        # Copy the file into a memory buffer - mainly so that we don't lock the file
        # on disk, but also this should theoretically improve performance when loading
        # random parts of the file.
        self._fname = fname
        self._zipfile = None

        if memory_buffer:
            with open(fname, "rb") as f:
                buffer = io.BytesIO(f.read())
            self._zipfile = zipfile.ZipFile(buffer, mode="r")

        # Read the identifiers
        with self.zipfile.open("identifiers.txt") as f:
            identifiers = f.readlines()
            identifiers = [x.decode().strip() for x in identifiers]

        with self.zipfile.open("summary.csv") as f:
            self.summary = pd.read_csv(f)
            self.summary.set_index(identifiers, inplace=True)
            self.summary = self.summary.sort_index()

        self._cache = {}  # Cache the dataframes

        if preload:
            self.preload()

    def copy(self):
        """
        Shallow copy - shared cache, copied summary

        This allows efficient filtering of seeds within runs by removing rows from
        the copy's summary, while not reloading or duplicating any of the dataframes
        in memory

        Returns:

        """

        new = copy.copy(self)  # Shallow copy of
        new.summary = copy.deepcopy(self.summary)
        return new

    def preload(self):
        """
        Load all dataframes into cache

        This is done based on the seeds in self.seeds, therefore if some of the seeds
        are removed prior to preloading, then those dataframes will not be loaded
        """

        _ = [self[x] for x in self.seeds]

    @property
    def zipfile(self):
        if self._zipfile:
            return self._zipfile
        else:
            return zipfile.ZipFile(self._fname, mode="r")

    def __repr__(self):
        return f"<Samples {'-'.join(str(x) for x in self.id.values())}, {len(self)} seeds>"

    @property
    def index(self):
        """Alias summary dataframe index"""
        return self.summary.index

    @property
    def columns(self):
        """Alias summary dataframe columns"""
        return self.summary.columns

    @property
    def identifier(self) -> namedtuple:
        """
        Return tuple identifier for this run

        The identifier is something like

            ('level_1',3,'level_5')

        Can be used to identify this result set in the context of sweeping over these identifiers
        e.g. the identifier could contain the starting level of restrictions. The dimensionality
        will vary depending on the analysis, hence it returns a tuple of arbitrary length. It would
        just be expected that all results being analyzed at the same time would have the same set of
        identifiers. The first two index levels are always 'beta' and 'seed', therefore these are
        dropped from the ID.

        """

        return tuple(self.summary.iloc[0].name[1:])

    @property
    def id(self) -> dict:
        """
        Return a dictionary with the identifiers and associated values

        For example:

        >>> result.id
        (2.0, 'Gradually escalate restrictions', 0.16, 0.5, '95_70', 1)
        >>>  result.identifier
        {'beta_multiplier': 2.0,
         'strategy': 'Gradually escalate restrictions',
         'symp_test': 0.16,
         'vac_rel_test': 0.5,
         'vac_peak_coverage': '95_70',
         'incursions_per_day': 1}

        Returns: A dictionary {identifier name: value}

        """
        return dict(zip(tuple(self.summary.index.names[1:]), self.identifier))

    def __len__(self) -> int:
        """
        Return number of runs

        Returns: Integer number of seeds

        """
        return len(self.summary)

    def __iter__(self):
        """
        Iterate over dataframes

        For example:

        >>> for df in result:
        >>>     max(df['new_diagnoses'])

        Returns: Generator over dataframes

        """
        for seed in self.seeds:
            yield self[seed]

    def __contains__(self, seed):
        """
        Check if a seed is contained
        Args:
            seed:

        Returns:

        """
        return seed in set(self.seeds)

    @property
    def seeds(self) -> np.array:
        """
        Return array of all seeds

        The seeds are 'registered' in the "seed" column of the summary dataframe.
        Therefore, to discard seeds, the rows can be dropped from the summary
        dataframe. In that case, iterating over `Samples.seeds()` will skip the excluded
        runs. Indexing into the Samples object will also fail to retrieve seeds that have
        been removed from the summary.

        Returns: Array of seed values

        """

        return self.index.get_level_values("seed").values

    @classmethod
    def new(cls, folder, outputs, identifiers, fname=None):
        """

        Args:
            fname:
            outputs: A list of tuples (df:pd.DataFrame, summary_row:dict) where the summary row as an entry 'seed' for the seed
            identifiers: A list of columns to use as identifiers. These should appear in the summary dataframe and should have the same
                         value for all samples.

        Returns: None

        """

        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, mode="x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:

            summary_rows = []
            for df, row in outputs:
                f = io.BytesIO()
                df.to_csv(f)
                f.flush()
                zf.writestr(f'seed_{row["seed"]}.csv', f.getvalue())
                summary_rows.append(row)

            # Write the identifier metadata
            if identifiers is None:
                identifiers = ["seed"]
            else:
                identifiers = ["seed"] + list(identifiers)
            identifiers = list(dict.fromkeys(identifiers))  # Unique, maintaining order

            # Write the summary
            summary = pd.DataFrame(summary_rows)

            # Check that each identifier column only contains one unique value
            for identifier in identifiers:
                if summary[identifier].nunique() > 1 and identifier != 'seed':
                    raise ValueError(f"Identifier column '{identifier}' contains more than one unique value")

            summary.set_index(identifiers, inplace=True)
            f = io.BytesIO()
            summary.to_csv(f)
            f.flush()
            zf.writestr(f"summary.csv", f.getvalue())

            zf.writestr("identifiers.txt", "\n".join(identifiers))

        buffer.flush()

        if fname is None:
            fname = "-".join(str(row[x]) for x in identifiers[1:]) + ".zip"
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / fname, mode="wb") as f2:
            f2.write(buffer.getvalue())

        return cls(folder / fname, memory_buffer=False)

    def get(self, seed) -> tuple:
        """
        Retrieve dataframe and summary row

        Use ``Samples[seed]`` to read only the dataframe.
        Use ``Samples.get(seed)`` to read both the dataframe and summary row

        Args:
            seed:

        Returns:

        """
        df = self[seed]
        row = sc.objdict(self.summary.xs(seed, level="seed").reset_index().iloc[0])
        return row, df

    def items(self) -> tuple:
        """
        Iterate over seeds and dataframes

        Example usage

        >>> res = Samples(...)
        >>> for seed, (row, df) in res:
        >>>     ...

        Returns: Tuple with
            - seed
            - Samples.get(seed) i.e. a tuple with
                - the summary dataframe row for the requested seed
                - the corresponding CSV output for that run

        """
        for seed in self.seeds:
            yield seed, self.get(seed)

    def __getitem__(self, item):
        """
        Overload getitem for convenience

        This provides two usages depending on the item

        - If the item is a number, it is treated as a seed
        - Otherwise, it is treated as a column in the summary dataframe
        - If the item is a two element tuple, the first element is a seed and the second element is a
          column in the individual run dataframe

        For example

        >>> a = results[0]
        >>> _, a = results.get(0)

        >>> a = results['cum_diagnoses']
        >>> a = results.summary['cum_diagnoses']

        >>> a = results[0,'new_diagnoses']
        >>> a = results[0]['new_diagnoses']

        Naturally, this means that the seeds must be integers (but this is the only data type
        that would make sense with Numpy's random number generator anyway)

        Args:
            item:

        Returns:

        """

        if isinstance(item, tuple):
            return self[item[0]][item[1]]

        if not sc.isnumber(item):
            if item not in self.columns:
                sc.suggest(item, self.columns, die=True)
            return self.summary[item]

        # Otherwise, assume the item is a seed
        if item not in self:
            raise Exception(f'This dataset does not contain item "{item}"')

        if item not in self._cache:
            with self.zipfile.open(f"seed_{item}.csv") as f:
                self._cache[item] = pd.read_csv(f, index_col="t")

        return self._cache[item].copy()

    def apply(self, fcn, *args, **kwargs) -> list:
        """
        Apply/map function to every dataframe

        The function will be applied to every individual dataframe in the collection.

        Args:
            fcn: A function to apply. It should take in a dataframe
            *args: Additional arguments for ``fcn``
            **kwargs: Additional arguments for ``fcn``

        Returns: A list with the output of ``fcn``

        """

        return [fcn(x, *args, **kwargs) for x in self]
