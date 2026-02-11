from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


from library import train_util
from train_control_net import setup_parser, train

if __name__ == "__main__":
    logger.warning(
        "The module 'train_controlnet.py' is deprecated. Please use 'train_control_net.py' instead"
        " / 'train_controlnet.py'は非推奨です。代わりに'train_control_net.py'を使用してください。"
    )
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
