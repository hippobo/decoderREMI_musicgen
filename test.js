

function remiTokensToMidi(tokens, ticksPerSample) {
    console.log(tokens)
    let timeDivision = 384;
    let ticksPerSample = timeDivision / 8;
    let track = new midiWriterJs.Track();

    let currentTick = 0;
    let tick_at_last_ts_change = 0;
    let tick_at_current_bar = 0;
    let currentBar = -1;
    let bar_at_last_ts_change = 0;
    let previous_note_end = 0;
    let currentProgram = isPianoOnly ? 0 : null; 
    let currentDuration = 'T120';
    let currentVelocity = 50;

        tokenGroup.forEach(token => {
            let [tokenType, tokenValue] = token.split('_');

            switch (tokenType) {


                case 'Position':
                    if (!isNaN(parseInt(tokenValue))) {
                        currentTick += parseInt(tokenValue) * ticksPerSample;
                    }
                case 'Pitch':                               
                    if (tokenValue !== null && !isNaN(tokenValue)) {

                        try{
                            vel_type, vel = tokenGroup[ti + 1].split("_")
                            dur_type, dur = tokenGroup[ti + 2].split("_")
                            if (vel_type == "Velocity" && dur_type == "Duration"){
                                let [beat, pos, res] = dur.split('.').map(Number);
                                let durationInTicks = (beat * res + pos) * timeDivision / res;
                                dur = 'T' + durationInTicks.toString();


                            const note = new midiWriterJs.NoteEvent({
                                pitch: [tokenValue],
                                startTick: currentTick,
                                endTick: currentTick + dur,
                                velocity: vel

                            });
                            
                            track.addEvent(note);
                            }


                        } catch(e) {
                            console.log(e)
                            break;
                        }

                        }
                    
                    break;
            
            
                    case 'Program':
                    if (!isPianoOnly) {
                        let programNumber = parseInt(tokenValue);
                        if (!isNaN(programNumber) && programNumber >= 0 && programNumber <= 127) {
                            currentProgram = programNumber;
                            track.addEvent(new midiWriterJs.ProgramChangeEvent({instrument: currentProgram}));
                        }
                    }
                    break;
               
                
            }
        });


    // Convert the track to a MIDI file
    let write = new midiWriterJs.Writer(track);
    return write.buildFile();
}