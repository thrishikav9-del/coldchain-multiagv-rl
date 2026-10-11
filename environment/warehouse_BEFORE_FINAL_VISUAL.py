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
    # A4 - EXTERNAL WEATHER STATE
    # ==========================================================

    def set_external_weather(
        self,
        temperature=25.0,
        humidity=60.0,
    ):
        """
        Set the external environmental weather conditions.

        A4:
            Weather is an external environmental input.
            It is intentionally kept separate from the
            warehouse zone thermal state.

        Parameters:
            temperature : external/outdoor temperature in Celsius
            humidity    : external relative humidity in percent
        """

        temperature = float(
            temperature
        )

        humidity = max(
            0.0,
            min(
                100.0,
                float(humidity),
            ),
        )

        object.__setattr__(
            self,
            "_external_weather",
            {
                "temperature": temperature,
                "humidity": humidity,
            },
        )

        return dict(
            self._external_weather
        )

    def get_external_weather(
        self,
    ):
        """
        Return the current external weather state.
        """

        if not hasattr(
            self,
            "_external_weather",
        ):
            self.set_external_weather()

        return dict(
            self._external_weather
        )

        # --------------------------------------------------
        # Goal-progress filtering
        # Keep collision-free actions that do not move an
        # AGV farther from its current operational target.
        # --------------------------------------------------

        if self._evaluation_env is not None:
            env = self._evaluation_env
            progress_indices = []

            for joint_index in safe_indices:
                actions = self.decode_joint_action(joint_index)
                makes_progress_or_stays = True

                for agv_id in range(1, self.num_agvs + 1):
                    agv = env.system.agvs.get(agv_id)
                    if agv is None:
                        makes_progress_or_stays = False
                        break

                    target = env._get_current_target(agv)
                    if target is None:
                        continue

                    current = tuple(map(int, agv.position))
                    proposed = tuple(map(int, env._get_target_position(agv, int(actions[agv_id - 1]))))

                    current_distance = abs(current[0] - int(target[0])) + abs(current[1] - int(target[1]))
                    proposed_distance = abs(proposed[0] - int(target[0])) + abs(proposed[1] - int(target[1]))

                    if proposed_distance > current_distance:
                        makes_progress_or_stays = False
                        break

                if makes_progress_or_stays:
                    progress_indices.append(joint_index)

            if progress_indices:
                safe_indices = progress_indices

        # --------------------------------------------------
    def fetch_external_weather(
        self,
        latitude: float,
        longitude: float,
        timeout: float = 10.0,
    ):
        """
        Fetch current outdoor weather from Open-Meteo.
        The returned temperature and humidity are stored through
        the existing set_external_weather() interface.
        """
        from urllib.parse import urlencode
        from urllib.request import urlopen
        import json
        latitude = float(latitude)
        longitude = float(longitude)
        timeout = float(timeout)
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}."
            )
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}."
            )
        if timeout <= 0.0:
            raise ValueError(
                "timeout must be positive."
            )
        params = urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m",
                "temperature_unit": "celsius",
                "timezone": "auto",
            }
        )
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            + params
        )
        try:
            with urlopen(
                url,
                timeout=timeout,
            ) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch external weather: {exc}"
            ) from exc
        current = payload.get(
            "current",
            {},
        )
        if "temperature_2m" not in current:
            raise RuntimeError(
                "Open-Meteo response missing current temperature_2m."
            )
        if "relative_humidity_2m" not in current:
            raise RuntimeError(
                "Open-Meteo response missing current relative_humidity_2m."
            )
        weather = self.set_external_weather(
            temperature=current["temperature_2m"],
            humidity=current["relative_humidity_2m"],
        )
        weather["latitude"] = latitude
        weather["longitude"] = longitude
        weather["time"] = current.get(
            "time",
            None,
        )
        return weather
    def update_zone_from_weather(
        self,
        zone_id,
        external_heat=0.0,
        cooling_effectiveness=None,
        power_available=None,
        thermal_gain=0.02,
        cooling_rate=0.05,
    ):
        """
        Advance a zone's thermal state using the current
        external weather conditions.

        Weather temperature is NOT the zone temperature.

        Instead, the external weather temperature is supplied
        to the existing simplified thermal model as the
        environmental boundary condition.
        """

        weather = self.get_external_weather()

        zone_id = int(
            zone_id
        )

        state = self.get_zone_thermal_state(
            zone_id
        )

        if cooling_effectiveness is None:
            cooling_effectiveness = state[
                "cooling_effectiveness"
            ]

        if power_available is None:
            power_available = state[
                "power_available"
            ]

        updated_state = self.update_zone_temperature(
            zone_id=zone_id,
            external_heat=external_heat,
            cooling_effectiveness=cooling_effectiveness,
            power_available=power_available,
            ambient_temperature=weather[
                "temperature"
            ],
            thermal_gain=thermal_gain,
            cooling_rate=cooling_rate,
        )

        # Weather humidity is an external condition.
        # For this simplified A4 model, the zone humidity
        # moves gradually toward outdoor humidity rather
        # than becoming equal to it immediately.

        updated_state = dict(
            updated_state
        )

        old_humidity = float(
            state.get(
                "humidity",
                50.0,
            )
        )

        external_humidity = float(
            weather[
                "humidity"
            ]
        )

        humidity_response_rate = 0.05

        new_humidity = (
            old_humidity
            +
            humidity_response_rate
            *
            (
                external_humidity
                -
                old_humidity
            )
        )

        updated_state[
            "humidity"
        ] = float(
            new_humidity
        )

        self._thermal_state[
            zone_id
        ][
            "humidity"
        ] = float(
            new_humidity
        )

        return dict(
            self._thermal_state[
                zone_id
            ]
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
        ambient_temperature=25.0,
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
        # ==========================================================
    # A5 - DERPG LOCAL ZONE RISK
    # ==========================================================

    def calculate_local_zone_risk(
        self,
        zone_id,
        safe_min_temperature=2.0,
        safe_max_temperature=8.0,
        trend_threshold=0.5,
    ):
        """
        Calculate the local environmental risk of one warehouse zone.

        This is the local component of the DERPG-style dynamic
        zone-risk model.

        Risk sources:

            1. Temperature outside the cold-chain safe range
            2. Temperature trend
            3. External heat
            4. Cooling degradation/failure
            5. Power failure

        Returns a normalized risk value in [0, 1].
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

        state = self.get_zone_thermal_state(
            zone_id
        )

        temperature = float(
            state.get(
                "temperature",
                4.0,
            )
        )

        temperature_trend = float(
            state.get(
                "temperature_trend",
                0.0,
            )
        )

        external_heat = max(
            0.0,
            float(
                state.get(
                    "external_heat",
                    0.0,
                )
            ),
        )

        cooling_effectiveness = max(
            0.0,
            min(
                1.0,
                float(
                    state.get(
                        "cooling_effectiveness",
                        1.0,
                    )
                ),
            ),
        )

        power_available = bool(
            state.get(
                "power_available",
                True,
            )
        )

        # ------------------------------------------------------
        # TEMPERATURE RISK
        # ------------------------------------------------------

        if temperature < safe_min_temperature:

            temperature_risk = min(
                1.0,
                (
                    safe_min_temperature
                    -
                    temperature
                )
                /
                max(
                    1.0,
                    safe_max_temperature
                    -
                    safe_min_temperature,
                ),
            )

        elif temperature > safe_max_temperature:

            temperature_risk = min(
                1.0,
                (
                    temperature
                    -
                    safe_max_temperature
                )
                /
                max(
                    1.0,
                    safe_max_temperature
                    -
                    safe_min_temperature,
                ),
            )

        else:

            temperature_risk = 0.0

        # ------------------------------------------------------
        # TEMPERATURE TREND RISK
        # ------------------------------------------------------

        trend_magnitude = abs(
            temperature_trend
        )

        if trend_threshold <= 0.0:

            trend_risk = min(
                1.0,
                trend_magnitude,
            )

        else:

            trend_risk = min(
                1.0,
                trend_magnitude
                /
                float(
                    trend_threshold
                ),
            )

        # ------------------------------------------------------
        # EXTERNAL HEAT RISK
        # ------------------------------------------------------

        external_heat_risk = min(
            1.0,
            external_heat
            /
            5.0,
        )

        # ------------------------------------------------------
        # COOLING RISK
        # ------------------------------------------------------

        cooling_risk = (
            1.0
            -
            cooling_effectiveness
        )

        # ------------------------------------------------------
        # POWER FAILURE RISK
        # ------------------------------------------------------

        power_risk = (
            0.0
            if power_available
            else 1.0
        )

        # ------------------------------------------------------
        # LOCAL RISK COMBINATION
        # ------------------------------------------------------

        local_risk = (
            0.50
            *
            temperature_risk
            +
            0.15
            *
            trend_risk
            +
            0.10
            *
            external_heat_risk
            +
            0.15
            *
            cooling_risk
            +
            0.10
            *
            power_risk
        )

        local_risk = max(
            0.0,
            min(
                1.0,
                float(
                    local_risk
                ),
            ),
        )

        return {
            "zone_id": zone_id,
            "temperature": temperature,
            "temperature_trend": temperature_trend,
            "external_heat": external_heat,
            "cooling_effectiveness": cooling_effectiveness,
            "power_available": power_available,
            "temperature_risk": float(
                temperature_risk
            ),
            "trend_risk": float(
                trend_risk
            ),
            "external_heat_risk": float(
                external_heat_risk
            ),
            "cooling_risk": float(
                cooling_risk
            ),
            "power_risk": float(
                power_risk
            ),
            "local_risk": float(
                local_risk
            ),
        }

    # ==========================================================
    # A5 - DERPG DYNAMIC ZONE RISK
    # ==========================================================

    def calculate_zone_risk(
        self,
        zone_id,
        propagation_weight=0.30,
    ):
        """
        Calculate the dynamic environmental risk of one zone.

        DERPG-style structure:

            Dynamic Risk
                =
            Local Risk
                +
            Neighbour Risk Propagation

        The current zone's local environmental risk is combined
        with the average risk of its directly adjacent zones.

        Risk is normalized to [0, 1].
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

        propagation_weight = max(
            0.0,
            min(
                1.0,
                float(
                    propagation_weight
                ),
            ),
        )

        local_data = (
            self.calculate_local_zone_risk(
                zone_id
            )
        )

        neighbors = (
            self.get_zone_neighbors(
                zone_id
            )
        )

        neighbor_risks = []

        for neighbor_id in neighbors:

            neighbor_data = (
                self.calculate_local_zone_risk(
                    neighbor_id
                )
            )

            neighbor_risks.append(
                float(
                    neighbor_data[
                        "local_risk"
                    ]
                )
            )

        if neighbor_risks:

            neighbor_risk = (
                sum(
                    neighbor_risks
                )
                /
                len(
                    neighbor_risks
                )
            )

        else:

            neighbor_risk = 0.0

        local_risk = float(
            local_data[
                "local_risk"
            ]
        )

        dynamic_risk = (
            (
                1.0
                -
                propagation_weight
            )
            *
            local_risk
            +
            propagation_weight
            *
            neighbor_risk
        )

        dynamic_risk = max(
            0.0,
            min(
                1.0,
                float(
                    dynamic_risk
                ),
            ),
        )

        return {
            "zone_id": zone_id,
            "local_risk": local_risk,
            "neighbor_risk": float(
                neighbor_risk
            ),
            "dynamic_risk": float(
                dynamic_risk
            ),
            "neighbors": list(
                neighbors
            ),
        }

    # ==========================================================
    # A5 - ALL ZONE RISKS
    # ==========================================================

    def get_all_zone_risks(
        self,
        propagation_weight=0.30,
    ):
        """
        Calculate dynamic environmental risk for every
        warehouse zone.
        """

        return {
            zone_id: self.calculate_zone_risk(
                zone_id=zone_id,
                propagation_weight=propagation_weight,
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

