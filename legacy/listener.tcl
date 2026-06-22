# listener.tcl - Non-blocking socket bridge for BRL-CAD
set port 5555

# End-of-response delimiter — Python reads lines until it sees this.
set EOR "<<END_OF_RESPONSE>>"

proc handle_client {sock addr client_port} {
    # Set the socket to non-blocking so your GUI doesn't freeze!
    fconfigure $sock -buffering line -blocking 0
    fileevent $sock readable [list process_command $sock]
    puts "MCP Server connected from $addr"
}

proc process_command {sock} {
    global EOR

    # If the Python script disconnects, close the socket
    if {[eof $sock]} {
        puts "MCP Server disconnected."
        close $sock
        return
    }

    # Read the command from Python
    if {[gets $sock cmd] >= 0} {
        puts "Executing AI Command: $cmd"
        
        # Use 'eval' so multi-command lines (semicolon-separated) work,
        # but wrap in a catch to prevent bad commands from crashing MGED.
        if {[catch {uplevel #0 $cmd} result]} {
            puts $sock "ERROR: $result"
        } else {
            puts $sock "SUCCESS: $result"
        }
        # Delimiter so Python knows the response is complete.
        puts $sock $EOR
        flush $sock
    }
}

# Start the server
socket -server handle_client $port
puts "================================================="
puts " BRL-CAD AI Bridge active."
puts " Listening for MCP commands on localhost:$port..."
puts "================================================="
