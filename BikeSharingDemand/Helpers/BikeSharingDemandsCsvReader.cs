using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BikeSharingDemand.BikeSharingDemandData;

namespace BikeSharingDemand.Helpers
{
    public class BikeSharingDemandsCsvReader
    {
        public IEnumerable<BikeSharingDemandSample> GetDataFromCsv(string dataLocation)
        {
            return File.ReadAllLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new BikeSharingDemandSample()
                {
                    Season = int.Parse(x[2]),
                    Year = int.Parse(x[3]),
                    Month = int.Parse(x[4]),
                    Hour = int.Parse(x[5]),
                    Holiday = float.Parse(x[6]) != 0,
                    Weekday = int.Parse(x[7]),
                    Weather = int.Parse(x[8]),
                    Temperature = float.Parse(x[9]),
                    NormalizedTemperature = float.Parse(x[10]),
                    Humidity = float.Parse(x[11]),
                    Windspeed = float.Parse(x[12]),
                    Count = float.Parse(x[15])
                });
        }
    }
}
