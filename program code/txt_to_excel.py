import pandas as pd

class txt_to_excel:
    def __init__(self, inp):
        self.inp = inp

        file_name = '{}.txt'.format(inp)
        print(file_name)

        new_file_name = '{}.xlsx'.format(inp)

        file = open(file_name, 'r')
        text = file.read()
        text_array = text.split(',')
        to_return = list()
        for i in range(len(text_array)-1):
            to_return.append(float(text_array[i]))

        df = pd.DataFrame({'Data': to_return})
        writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        print('completed')

if __name__ == '__main__':
    print('start txt to excel')
    inp = str(input())
    while inp is not 'quit':
        txt_to_excel(inp)
        inp = str(input())


