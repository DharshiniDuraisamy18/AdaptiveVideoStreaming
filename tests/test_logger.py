from logger import initialize_log, log_event


initialize_log()


log_event(
    activity="IDLE",
    tier=0,
    width=320,
    height=240,
    fps=5,
    motion=0.2,
    bytes_sent=15000
)


log_event(
    activity="PERSON",
    tier=2,
    width=640,
    height=480,
    fps=10,
    motion=3.5,
    bytes_sent=50000
)


print("Logging test completed.")