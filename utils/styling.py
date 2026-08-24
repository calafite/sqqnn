class ColourCodes:
    reset: str = "\033[0m"
    bold: str = "\033[1m"
    red: str = "\033[31m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    blue: str = "\033[34m"
    magenta: str = "\033[35m"
    cyan: str = "\033[36m"
    white: str = "\033[37m"


class ColouredString:
    def __init__(self, text: str, codes: list[str] | None = None) -> None:
        self.raw_text: str = text
        self.applied_codes: list[str] = codes or []

    def _chain(self, code: str) -> "ColouredString":
        new_codes: list[str] = self.applied_codes + [code]
        return ColouredString(self.raw_text, new_codes)

    def red(self) -> "ColouredString":
        return self._chain(ColourCodes.red)

    def green(self) -> "ColouredString":
        return self._chain(ColourCodes.green)

    def yellow(self) -> "ColouredString":
        return self._chain(ColourCodes.yellow)

    def blue(self) -> "ColouredString":
        return self._chain(ColourCodes.blue)

    def magenta(self) -> "ColouredString":
        return self._chain(ColourCodes.magenta)

    def cyan(self) -> "ColouredString":
        return self._chain(ColourCodes.cyan)

    def white(self) -> "ColouredString":
        return self._chain(ColourCodes.white)

    def bold(self) -> "ColouredString":
        return self._chain(ColourCodes.bold)

    def __str__(self) -> str:
        if not self.applied_codes:
            return self.raw_text
        prefix: str = "".join(self.applied_codes)
        return f"{prefix}{self.raw_text}{ColourCodes.reset}"


def colourise(target: object) -> ColouredString:
    return ColouredString(str(target))


class OutputFormatter:
    @staticmethod
    def title(prompt: str, width: int = 45) -> None:
        title_str: str = f"[{prompt.upper()}]".center(width, " ")
        styled: ColouredString = colourise(title_str).cyan().bold()
        print(f"\n{styled}")

    @staticmethod
    def config(name: str, value: object, indent: int = 12) -> None:
        pad_space: str = " " * indent
        key_str: ColouredString = colourise(name).white()
        val_str: ColouredString = colourise(str(value)).cyan()

        print(f"{pad_space}* {key_str}: {val_str}")

    @staticmethod
    def metric(name: str, value: float | str, indent: int = 12) -> None:
        pad_space: str = " " * indent
        key_str: ColouredString = colourise(name).white()

        value_fmt: str = f"{value:.4f}" if isinstance(value, float) else str(value)
        val_str: ColouredString = colourise(value_fmt).green().bold()

        print(f"{pad_space}  {key_str}: {val_str}")
