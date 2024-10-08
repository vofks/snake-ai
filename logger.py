import csv
import os
import datetime

RESULT_FOLDER = 'results'
FILE_FORMAT = '.csv'


class ExperimentLog:
    def __init__(self, project_name, timestamp):
        self.filename = project_name + '_' + \
            timestamp.strftime('%H-%M-%S %d-%m-%Y') + FILE_FORMAT

    def setup(self):
        if not os.path.exists(RESULT_FOLDER):
            os.mkdir(RESULT_FOLDER)

        self.path = os.path.join(RESULT_FOLDER, self.filename)

        if os.path.exists(self.path):
            print(
                f'Project log error. File: {self.path} already exists. Abort.')
            return False

        return True

    def logrow(self, data):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(data)


if __name__ == '__main__':
    log = ExperimentLog('test', datetime.datetime.now())
    success = log.setup()

    if not success:
        print('Error')

    log.logrow([123, 2222, 12, 10000, datetime.datetime.now()])
    log.logrow([123, 2222, 12, 10000, datetime.datetime.now()])
    log.logrow([123, 2222, 12, 10000, datetime.datetime.now()])
