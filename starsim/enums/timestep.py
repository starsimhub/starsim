from enum import Enum

__all__ = ['TimeStep']

class TimeStep(Enum):
    """
    An enumeration representing different time steps with their corresponding values in years.
    
    Enum Members:   DAY | WEEK | MONTH | YEAR
        DAY (float): Represents one day as a fraction of a year (1/365).
        WEEK (float): Represents one week as a fraction of a year (7/365).
        MONTH (float): Represents one month as a fraction of a year (30.41666666666667/365).
        YEAR (float): Represents one year (1.0).

    Properties:
        describe (tuple): A property that returns a tuple containing the name and value of the enum member.

    Methods:
        Name() -> str: Returns the name of the enum member.
        __repr__() -> str: Returns a string representation of the enum member, which is its name.
        __float__() -> float: Returns the value of the enum member as a float.
        __str__() -> str: Returns a string representation of the value of the enum member.
        
    Examples:    
        
        from starsim import TimeStep

        # Example: Using the describe property
        print(TimeStep.DAY.describe)  # Output: ('DAY', 0.0027397260273972603)
        
        # Example: Using the Name method
        print(TimeStep.DAY.Name())  # Output: 'DAY'
        
        # Example: Accessing enum members
        print(TimeStep.DAY)  # Output: 0.0027397260273972603

        # Example: Getting the value as a float
        print(float(TimeStep.DAY))  # Output: 0.0027397260273972603

        # Example: Getting the string representation of the value
        print(str(TimeStep.DAY))  # Output: '0.0027397260273972603'

    """
    _order_ = 'DAY WEEK MONTH YEAR'
    DAY = 1/365
    WEEK = 7/365
    MONTH = 30.41666666666667/365
    YEAR = 1.0
   
    @property
    def describe(self):
        """
        Provides a tuple containing the name and value of the enum member.

        Returns:
            tuple: A tuple containing the name and value of the enum member.
        """
        return self.name, self.value
    
    def Name(self):
        """
        Returns the name of the enum member.

        Returns:
            str: The name of the enum member.
        """
        return self.name
    
    def __repr__(self):
        """
        Returns a string representation of the enum member.

        Returns:
            str: The name of the enum member.
        """
        return '{0}'.format(self.name)
    
    def __float__(self):
        """
        Returns the value of the enum member as a float.

        Returns:
            float: The value of the enum member.
        """
        return self.value
    
    def __str__(self):
        """
        Returns a string representation of the value of the enum member.

        Returns:
            str: The value of the enum member as a string.
        """
        return '{0}'.format(self.value)


if __name__ == '__main__':
    help(TimeStep)

