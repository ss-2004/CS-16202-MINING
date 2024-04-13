import pandas as pd

Columns_list = ['Reg_no', 'Name', 'Subject1', 'Subject2', 'Subject3', 'Subject4']
Rows_list = [
    [2022001, 'Abhijeet', 65, 65, 69, 81],
    [2022002, 'Ajeet', 75, 75, 90, 81],
    [2022003, 'Amit', 75, 55, 69, 87],
    [2022004, 'Ranjeet', 55, 65, 79, 91],
    [2022005, 'Santosh', 85, 85, 60, 61],
    [2022006, 'Satyam', 73, 75, 68, 51],
    [2022007, 'Shivam', 85, 85, 50, 40],
    [2022009, 'Shyam', 75, 65, 69, 81],
    [2022010, 'Yash', 85, 75, 89, 61]
]

df = pd.DataFrame(Rows_list, columns=Columns_list)
df['Total'] = df['Subject1'] + df['Subject2'] + df['Subject3'] + df['Subject4']

def get_grade(total):
    if total >= 90:
        return 'A'
    elif total >= 80:
        return 'B'
    elif total >= 70:
        return 'C'
    elif total >= 50:
        return 'D'
    else:
        return 'E'

df['Grade'] = df['Total'].apply(get_grade)
subset = df[['Reg_no', 'Name', 'Grade']]
subset.to_csv('students_grade.csv', index=False)
print(df)
