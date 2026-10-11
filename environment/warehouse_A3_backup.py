"""
Grid-based warehouse for the multi-AGV cold-chain system.

The warehouse provides:

    - dimensions
    - obstacles
    - traversable cells
    - start/depot cells
    - goal cells
    - spatial zones

Phase A1:
    Add spatial warehouse zones.

Important:
    Zones are purely spatial in this phase.

    Temperature, cooling, weather, thermal propagation,
    product exposure and risk are implemented in later
    cold-chain phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Position = tuple[int, int]


@dataclass(frozen=True)
class Warehouse:

    width: int = 20
    height: int = 20

    # ----------------------------------------------------------
    # Phase A1 zone configuration
    # ----------------------------------------------------------

    zone_rows: int = 4
    zone_cols: int = 4

    def __post_init__(self):

        # ======================================================
        # BASIC VALIDATION
        # ======================================================

        if self.width <= 0:

            raise ValueError(
                "Warehouse width must be positive."
            )

        if self.height <= 0:

            raise ValueError(
                "Warehouse height must be positive."
            )

        if self.zone_rows <= 0:

            raise ValueError(
                "zone_rows must be positive."
            )

        if self.zone_cols <= 0:

            raise ValueError(
                "zone_cols must be positive."
            )

        # ======================================================
        # OBSTACLES
        # ======================================================

        object.__setattr__(
            self,
            "_obstacles",
            set(),
        )

        # ======================================================
        # ZONE DIMENSIONS
        # ======================================================

        # ceil(width / zone_cols)
        zone_width = (
            self.width
            + self.zone_cols
            - 1
        ) // self.zone_cols

        zone_height = (
            self.height
            + self.zone_rows
            - 1
        ) // self.zone_rows

        object.__setattr__(
            self,
            "_zone_width",
            zone_width,
        )

        object.__setattr__(
            self,
            "_zone_height",
            zone_height,
        )

    # ==========================================================
    # BOUNDS
    # ==========================================================

    def is_inside(
        self,
        position: Position,
    ) -> bool:

        x, y = position

        return (
            0 <= x < self.width
            and
            0 <= y < self.height
        )

    # ==========================================================
    # OBSTACLES
    # ==========================================================

    def add_obstacle(
        self,
        position: Position,
    ) -> None:

        if not self.is_inside(position):

            raise ValueError(
                f"Obstacle {position} "
                "is outside warehouse."
            )

        self._obstacles.add(
            position
        )

    def add_obstacles(
        self,
        positions: Iterable[Position],
    ) -> None:

        for position in positions:

            self.add_obstacle(
                position
            )

    def remove_obstacle(
        self,
        position: Position,
    ) -> None:

        self._obstacles.discard(
            position
        )

    def is_obstacle(
        self,
        position: Position,
    ) -> bool:

        return (
            position
            in
            self._obstacles
        )

    # ==========================================================
    # TRAVERSABILITY
    # ==========================================================

    def is_free(
        self,
        position: Position,
    ) -> bool:

        return (
            self.is_inside(position)
            and
            not self.is_obstacle(position)
        )

    # ==========================================================
    # NEIGHBOURS
    # ==========================================================

    def neighbors(
        self,
        position: Position,
    ) -> list[Position]:

        x, y = position

        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        return [
            candidate
            for candidate in candidates
            if self.is_free(candidate)
        ]

    # ==========================================================
    # OBSTACLE ACCESS
    # ==========================================================

    @property
    def obstacles(
        self,
    ) -> frozenset[Position]:

        return frozenset(
            self._obstacles
        )

    # ==========================================================
    # ZONE HELPERS
    # ==========================================================

    @property
    def total_zones(
        self,
    ) -> int:

        return (
            self.zone_rows
            *
            self.zone_cols
        )

    @property
    def zone_width(
        self,
    ) -> int:

        return self._zone_width

    @property
    def zone_height(
        self,
    ) -> int:

        return self._zone_height

    # ==========================================================
    # POSITION -> ZONE
    # ==========================================================

    def get_zone_id(
        self,
        position: Position,
    ) -> int:
        """
        Return the zone ID containing a warehouse position.

        Zone numbering is row-major from the bottom-left:

            1   2   3   4
            5   6   7   8
            9  10  11  12
           13  14  15  16

        Returns:
            Integer zone ID starting from 1.
        """

        if not self.is_inside(
            position
        ):

            raise ValueError(
                f"Position {position} "
                "is outside warehouse."
            )

        x, y = position

        zone_col = min(
            x // self.zone_width,
            self.zone_cols - 1,
        )

        zone_row = min(
            y // self.zone_height,
            self.zone_rows - 1,
        )

        return (
            zone_row
            *
            self.zone_cols
            +
            zone_col
            +
            1
        )

    # ==========================================================
    # ZONE -> BOUNDS
    # ==========================================================

    def get_zone_bounds(
        self,
        zone_id: int,
    ) -> tuple[int, int, int, int]:
        """
        Return:

            (x_min, x_max, y_min, y_max)

        for one zone.

        Bounds are inclusive.
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        zone_index = (
            zone_id - 1
        )

        zone_row = (
            zone_index
            //
            self.zone_cols
        )

        zone_col = (
            zone_index
            %
            self.zone_cols
        )

        x_min = (
            zone_col
            *
            self.zone_width
        )

        x_max = min(
            x_min
            + self.zone_width
            - 1,
            self.width - 1,
        )

        y_min = (
            zone_row
            *
            self.zone_height
        )

        y_max = min(
            y_min
            + self.zone_height
            - 1,
            self.height - 1,
        )

        return (
            x_min,
            x_max,
            y_min,
            y_max,
        )

    # ==========================================================
    # ZONE CELLS
    # ==========================================================

    def get_zone_cells(
        self,
        zone_id: int,
    ) -> list[Position]:
        """
        Return all cells belonging to a zone.
        """

        (
            x_min,
            x_max,
            y_min,
            y_max,
        ) = self.get_zone_bounds(
            zone_id
        )

        return [
            (x, y)
            for y in range(
                y_min,
                y_max + 1,
            )
            for x in range(
                x_min,
                x_max + 1,
            )
        ]

    # ==========================================================
    # ZONE CENTER
    # ==========================================================

    def get_zone_center(
        self,
        zone_id: int,
    ) -> Position:
        """
        Return an approximate center cell for a zone.
        """

        (
            x_min,
            x_max,
            y_min,
            y_max,
        ) = self.get_zone_bounds(
            zone_id
        )

        return (
            (x_min + x_max) // 2,
            (y_min + y_max) // 2,
        )

    # ==========================================================
    # POSITION -> ZONE INFORMATION
    # ==========================================================

    def get_zone_info(
        self,
        zone_id: int,
    ) -> dict:
        """
        Return basic spatial metadata for a zone.

        No thermal/risk information is included yet.
        """

        zone_id = int(
            zone_id
        )

        bounds = (
            self.get_zone_bounds(
                zone_id
            )
        )

        return {
            "zone_id": zone_id,
            "bounds": bounds,
            "center": self.get_zone_center(
                zone_id
            ),
            "cell_count": len(
                self.get_zone_cells(
                    zone_id
                )
            ),
        }

    # ==========================================================
    # ALL ZONES
    # ==========================================================

    def get_all_zone_info(
        self,
    ) -> list[dict]:
        """
        Return metadata for every warehouse zone.
        """

        return [
            self.get_zone_info(
                zone_id
            )
            for zone_id in range(
                1,
                self.total_zones + 1,
            )
        ]

    # ==========================================================
    # ZONE NEIGHBOURS
    # ==========================================================

    def get_zone_neighbors(
        self,
        zone_id: int,
    ) -> list[int]:
        """
        Return spatially adjacent zone IDs.

        Only four-neighbour adjacency is used:

            left
            right
            up
            down

        This will later provide the spatial structure
        required by DERPG.
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        zone_index = (
            zone_id - 1
        )

        row = (
            zone_index
            //
            self.zone_cols
        )

        col = (
            zone_index
            %
            self.zone_cols
        )

        neighbors = []

        # ------------------------------------------------------
        # LEFT
        # ------------------------------------------------------

        if col > 0:

            neighbors.append(
                zone_id - 1
            )

        # ------------------------------------------------------
        # RIGHT
        # ------------------------------------------------------

        if (
            col
            <
            self.zone_cols - 1
        ):

            neighbors.append(
                zone_id + 1
            )

        # ------------------------------------------------------
        # DOWN
        # ------------------------------------------------------

        if row > 0:

            neighbors.append(
                zone_id
                -
                self.zone_cols
            )

        # ------------------------------------------------------
        # UP
        # ------------------------------------------------------

        if (
            row
            <
            self.zone_rows - 1
        ):

            neighbors.append(
                zone_id
                +
                self.zone_cols
            )

        return neighbors
    # ==========================================================
    # VISUAL REPRESENTATION
    # ==========================================================

    def render_ascii(
        self,
        agv_positions: dict[int, Position] | None = None,
        goals: Iterable[Position] | None = None,
    ) -> str:

        agv_positions = (
            agv_positions or {}
        )

        goals = set(
            goals or []
        )

        agv_lookup = {
            position: agv_id
            for agv_id, position
            in agv_positions.items()
        }

        rows = []

        for y in reversed(
            range(
                self.height
            )
        ):

            row = []

            for x in range(
                self.width
            ):

                position = (
                    x,
                    y,
                )

                if position in agv_lookup:

                    row.append(
                        str(
                            agv_lookup[
                                position
                            ]
                        )
                    )

                elif position in self._obstacles:

                    row.append(
                        "#"
                    )

                elif position in goals:

                    row.append(
                        "G"
                    )

                else:

                    row.append(
                        "."
                    )

            rows.append(
                " ".join(row)
            )

        return "\n".join(
            rows
        )
        
        # ==========================================================
    # THERMAL STATE — PHASE A2
    # ==========================================================

    def initialize_thermal_state(
        self,
        default_temperature=4.0,
        default_humidity=50.0,
    ):
        """
        Initialize the runtime thermal state of every zone.

        Phase A2 only:
            - temperature
            - previous temperature
            - temperature trend
            - humidity
            - external heat
            - cooling effectiveness
            - power availability

        This is a simplified simulation model and is not
        intended to represent a physically calibrated warehouse.
        """

        default_temperature = float(
            default_temperature
        )

        default_humidity = float(
            default_humidity
        )

        thermal_state = {}

        for zone_id in range(
            1,
            self.total_zones + 1,
        ):

            thermal_state[zone_id] = {
                "zone_id": int(zone_id),

                "temperature": (
                    default_temperature
                ),

                "previous_temperature": (
                    default_temperature
                ),

                "temperature_trend": 0.0,

                "humidity": (
                    default_humidity
                ),

                "external_heat": 0.0,

                "cooling_effectiveness": 1.0,

                "power_available": True,
            }

        object.__setattr__(
            self,
            "_thermal_state",
            thermal_state,
        )

        return {
            zone_id: dict(state)
            for zone_id, state
            in thermal_state.items()
        }

    # ==========================================================
    # GET THERMAL STATE
    # ==========================================================

    def get_zone_thermal_state(
        self,
        zone_id,
    ):
        """
        Return a copy of one zone's current thermal state.
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        if not hasattr(
            self,
            "_thermal_state",
        ):

            self.initialize_thermal_state()

        return dict(
            self._thermal_state[
                zone_id
            ]
        )

    # ==========================================================
    # GET ALL THERMAL STATES
    # ==========================================================

    def get_all_zone_thermal_states(
        self,
    ):
        """
        Return copies of all zone thermal states.
        """

        if not hasattr(
            self,
            "_thermal_state",
        ):

            self.initialize_thermal_state()

        return {
            zone_id: dict(state)
            for zone_id, state
            in self._thermal_state.items()
        }

    # ==========================================================
    # UPDATE ONE ZONE TEMPERATURE
    # ==========================================================

    def update_zone_temperature(
        self,
        zone_id,
        external_heat=0.0,
        cooling_effectiveness=1.0,
        power_available=True,
        ambient_temperature=self.weather_temperature
        thermal_gain=0.02,
        cooling_rate=0.05,
    ):
        """
        Advance one zone's simplified thermal state by one
        simulation step.

        Conceptual model:

            environmental effect
            +
            external heat
            -
            cooling effect

        Temperature trend is:

            T(t+1) - T(t)

        This is intentionally simple at Phase A2.
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        if not hasattr(
            self,
            "_thermal_state",
        ):

            self.initialize_thermal_state()

        state = self._thermal_state[
            zone_id
        ]

        old_temperature = float(
            state[
                "temperature"
            ]
        )

        external_heat = float(
            external_heat
        )

        cooling_effectiveness = max(
                0.0,
    min(
        1.0,
        float(
            cooling_effectiveness
        ),
    )
)

        power_available = bool(
            power_available
        )

        ambient_temperature = float(
            ambient_temperature
        )

        thermal_gain = max(
            0.0,
            float(
                thermal_gain
            ),
        )

        cooling_rate = max(
            0.0,
            float(
                cooling_rate
            ),
        )

        environmental_term = (
            thermal_gain
            *
            (
                ambient_temperature
                -
                old_temperature
            )
        )

        if power_available:

            cooling_term = (
                cooling_rate
                *
                cooling_effectiveness
            )

        else:

            cooling_term = 0.0

        new_temperature = (
            old_temperature
            +
            environmental_term
            +
            external_heat
            -
            cooling_term
        )

        temperature_trend = (
            new_temperature
            -
            old_temperature
        )

        state[
            "previous_temperature"
        ] = old_temperature

        state[
            "temperature"
        ] = float(
            new_temperature
        )

        state[
            "temperature_trend"
        ] = float(
            temperature_trend
        )

        state[
            "external_heat"
        ] = float(
            external_heat
        )

        state[
            "cooling_effectiveness"
        ] = float(
            cooling_effectiveness
        )

        state[
            "power_available"
        ] = bool(
            power_available
        )

        return dict(
            state
        )

    # ==========================================================
    # RESET ALL THERMAL STATES
    # ==========================================================

    def reset_thermal_state(
        self,
        default_temperature=4.0,
        default_humidity=50.0,
    ):
        """
        Reset every zone to the default thermal condition.
        """

        return self.initialize_thermal_state(
            default_temperature=default_temperature,
            default_humidity=default_humidity,
        )
        
        # ==========================================================
    # COOLING CONTROL — PHASE A3
    # ==========================================================

    def set_zone_cooling(
        self,
        zone_id,
        effectiveness=1.0,
    ):
        """
        Set the cooling effectiveness of one zone.

        Range:

            1.0 -> normal cooling
            0.5 -> degraded cooling
            0.0 -> cooling unavailable
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        if not hasattr(
            self,
            "_thermal_state",
        ):

            self.initialize_thermal_state()

        effectiveness = max(
            0.0,
            min(
                1.0,
                float(
                    effectiveness
                ),
            ),
        )

        self._thermal_state[
            zone_id
        ][
            "cooling_effectiveness"
        ] = effectiveness

        return self.get_zone_thermal_state(
            zone_id
        )

    # ==========================================================
    # POWER CONTROL
    # ==========================================================

    def set_zone_power(
        self,
        zone_id,
        power_available=True,
    ):
        """
        Set electrical power availability for one zone.
        """

        zone_id = int(
            zone_id
        )

        if (
            zone_id < 1
            or
            zone_id > self.total_zones
        ):

            raise ValueError(
                f"Invalid zone_id {zone_id}. "
                f"Expected 1..{self.total_zones}."
            )

        if not hasattr(
            self,
            "_thermal_state",
        ):

            self.initialize_thermal_state()

        self._thermal_state[
            zone_id
        ][
            "power_available"
        ] = bool(
            power_available
        )

        return self.get_zone_thermal_state(
            zone_id
        )

    # ==========================================================
    # COOLING FAILURE EVENT
    # ==========================================================

    def trigger_cooling_failure(
        self,
        zone_id,
    ):
        """
        Simulate a complete cooling failure.

        Effects:

            cooling_effectiveness = 0
            power_available = False

        The temperature itself is not changed immediately.
        It changes through subsequent thermal updates.
        """

        state = self.set_zone_cooling(
            zone_id,
            effectiveness=0.0,
        )

        self.set_zone_power(
            zone_id,
            power_available=False,
        )

        return self.get_zone_thermal_state(
            zone_id
        )

    # ==========================================================
    # COOLING RECOVERY EVENT
    # ==========================================================

    def restore_cooling(
        self,
        zone_id,
        effectiveness=1.0,
    ):
        """
        Restore cooling after a simulated failure.
        """

        self.set_zone_power(
            zone_id,
            power_available=True,
        )

        self.set_zone_cooling(
            zone_id,
            effectiveness=effectiveness,
        )

        return self.get_zone_thermal_state(
            zone_id
        )

    # ==========================================================
    # COOLING STATUS
    # ==========================================================

    def get_zone_cooling_status(
        self,
        zone_id,
    ):
        """
        Return a simplified cooling status label.

        NORMAL
        DEGRADED
        FAILED
        """

        state = self.get_zone_thermal_state(
            zone_id
        )

        effectiveness = float(
            state[
                "cooling_effectiveness"
            ]
        )

        power_available = bool(
            state[
                "power_available"
            ]
        )

        if (
            not power_available
            or
            effectiveness <= 0.0
        ):

            status = "FAILED"

        elif effectiveness < 0.999:

            status = "DEGRADED"

        else:

            status = "NORMAL"

        return {
            "zone_id": int(
                zone_id
            ),
            "status": status,
            "cooling_effectiveness": effectiveness,
            "power_available": power_available,
        }

    # ==========================================================
    # ALL COOLING STATUS
    # ==========================================================

    def get_all_cooling_status(
        self,
    ):
        """
        Return cooling status for all zones.
        """

        return {
            zone_id: self.get_zone_cooling_status(
                zone_id
            )
            for zone_id in range(
                1,
                self.total_zones + 1,
            )
        }


__all__ = [
    "Warehouse",
    "Position",
]

