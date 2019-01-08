import argparse
from api.index import create_app
from api.model import train_monthly, train_daily

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload csv to database')
    parser.add_argument('-m', dest='mode', action='store',
                        help='train daily model or monthly model',
                        choices=['m', 'd'],
                        default='m')
    parser.add_argument('-t', dest='tenant_id', action='store',
                        help='tenant_id of the model',
                        default='0')
    args = parser.parse_args()
    print('Start training. Mode: {}, Tenant id: {}'.format(args.mode, args.tenant_id))
    app = create_app()
    with app.app_context():
        if args.mode == 'm':
            train_monthly(args.tenant_id)
        else:
            train_daily(args.tenant_id)
