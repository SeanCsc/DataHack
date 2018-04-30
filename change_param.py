base_total_expen = fairfield['TotalExpenditure15-16']
college_entrance_pct_2014 = fairfield['CollegeEntrancePercentage(2013-14)']
for x in range(300):
    fairfield['TotalExpenditure15-16'] = base_total_expen * (1 + .05 * x)
    fairfield['PercentChangeinExpenditure05-16'] = ((fairfield['TotalExpenditure15-16'] - fairfield['TotalExpenditure05-06']) / fairfield['TotalExpenditure05-06']) * 100
    
    fairfield['CollegeEntrancePercentage(2013-14)'] = college_entrance_pct_2014 * (1 + .05 * x)
    fairfield['PercentChangeinCollegeEntrancePercentage'] = ((fairfield['CollegeEntrancePercentage(2013-14)'] - fairfield['TotalExpenditure05-06']) / fairfield['TotalExpenditure05-06']) * 100
    print(xgb.predict(fairfield))