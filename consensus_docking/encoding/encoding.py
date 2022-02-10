import pandas as pd


class Encoding(object):
    """Encoding object"""
    def __init__(self):
        """
        It initializes a ParserEncoding object.
        """
        self.df = None

    @staticmethod
    def from_csv(encoding_file): 
        self.df = pd.read_csv(encoding_file, header=0)


    @staticmethod
    def from_dataframe(df): 
        self.df = df


    def __add_program_col(self):
        """
        Adds a column to self.df with the name of the program (based on the ids column) that obtained that pose.
        """
        self.df['program'] = self.df['ids'].str.split('_').str[0]

    @staticmethod
    def __flatten_list(lst):
        """
        Turns multidimensional list in a 1D list.
        Parameters
        ----------
        lst : list
            List to be flatten

        Returns
        -------
        Flattened list
        """
        from itertools import chain
        return list(chain.from_iterable(lst))

    @staticmethod
    def __select_df_col(df, selected_columns):
        """
        Verifies if the selected_columns belong to the input DataFrames.columns and raises an error if they don't belong
        to DataFrames.columns, otherwise, it selects selected_columns and returns the filtered DataFrame.
        Parameters
        ----------
        df : pandas DataFrame

        selected_columns : list
            List with the names of the columns we want to select.

        Returns
        -------
        pandas DataFrame with the desired columns.
        """
        if set(selected_columns).issubset(df.columns):
            return df[selected_columns]
        else:
            raise KeyError(
                f'Your selected column names ({selected_columns}) mismatch with the ones in the DataFrame'
                f' ({df.columns}).')

    def get_columns_by_name(self, selected_columns):
        """
        Parses the encoding csv file and returns a DataFrame with the selected columns.
        Parameters
        ----------
        selected_columns : list
            List with the names of the columns we want to select.

        Returns
        -------
        pandas DataFrame with the desired columns.
        """
        self.parse()
        return self.__select_df_col(self.df, selected_columns)

    def get_coord(self):
        """
        Gets the coordinate columns and returns them.

        Returns
        -------
        pandas DataFrame with the coordinate columns.
        """
        selected_columns = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3']
        self.parse()
        return self.__select_df_col(self.df, selected_columns)

    def get_coord_norm_sc(self):
        """
        Gets the coordinate and norm_score columns and returns them.

        Returns
        -------
        pandas DataFrame with the coordinate and norm_score columns.
        """
        selected_columns = ['norm_score', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3']
        self.parse()
        return self.__select_df_col(self.df, selected_columns)

    def get_ids(self):
        """
        Gets the ids column and returns it.

        Returns
        -------
        pandas DataFrame with the coordinate and norm_score columns.
        """
        selected_columns = ['ids']
        self.parse()
        return self.__select_df_col(self.df, selected_columns)

    def get_ids_by_row(self, index_list):
        """
        Returns list that has that follows the same order that appears in index_list
        """
        ids_df = self.get_ids()
        out_df = ids_df.iloc[index_list]
        out_list = out_df.values.tolist()
        return self.__flatten_list(out_list)

    def get_program_weights(self):
        #TODO alguna funcio que ponderi tenint en compte quantes poses te cada programa
        # (per exemple)
        # si ftdock te 10k i zdock te 2k --> les poses de ftdock valen 1 i les de zdock 5
        # La idea seria fer que el que té mes estructures tingui pes 1 i la resta anar mutiplicant.
        # Pero amb aquest approach s'hauria de controlar que es considera minim de poblacio (min_sample).
        pass
